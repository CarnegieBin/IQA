import os
import glob
import sys
import torch
import json
import math
import time
import logging
import numpy as np
import random
import argparse
import torch.distributed as dist
from torch.nn import functional as F
os.environ['USE_WKV_CUDA_FOR_RWKV'] = 'True'
from tqdm import tqdm
from pathlib import Path
from util.slconfig import DictAction, SLConfig
from util.logger import setup_logger
from util import misc
from util.utils import BestMetricHolder
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from dataset.dataset import TransformerXLTrainDataSet, TransformerXLTestDataSet
from dataset.enwik_dataset import EnWikTrainDataSet, EnWikTestDataSet
from dataset.enwik_ascii_dataset import EnWikASCIITrainDataSet, EnWikASCIITestDataSet
from util.cal import calculate_metrics as cal


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformerXL predictor', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # training parameters
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1204, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument("--distributed", default=True, action='store_true')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def get_word2id_dict(dict_path):
    with open(dict_path, "r") as obj_f:
        word2id_dict = json.load(obj_f)
    return word2id_dict


def get_ascii_word2id_dict():
    word2id_dict = {}
    word2id_dict['<s>'] = len(word2id_dict)
    word2id_dict['<unk>'] = len(word2id_dict)
    word2id_dict['<pad>'] = len(word2id_dict)
    
    byte_order = 'big'
    for i in range(256):
        byte_val = i.to_bytes(1, byte_order)
        word2id_dict[byte_val] = len(word2id_dict)
    return word2id_dict


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.model_name in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.model_name)
    model, criterion = build_func(args)
    return model, criterion


def print_param_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def evaluate(model, test_loader, rank, logger, save_path, best=None):
    world_size = dist.get_world_size()
    authentic = []
    prediction = []

    with torch.no_grad():
        model.eval()
        for image, mos in tqdm(test_loader):
            image = image.to(rank)
            mos = mos.to(rank)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(image)
            authentic.append(mos.detach().cpu())
            prediction.append(pred.detach().cpu())

    authentic_tensor = torch.cat(authentic)
    prediction_tensor = torch.cat(prediction)

    gathered_authentic = [torch.zeros_like(authentic_tensor) for _ in range(world_size)]
    gathered_prediction = [torch.zeros_like(prediction_tensor) for _ in range(world_size)]

    dist.all_gather(gathered_authentic, authentic_tensor)
    dist.all_gather(gathered_prediction, prediction_tensor)

    all_authentic = torch.cat(gathered_authentic).numpy()
    all_prediction = torch.cat(gathered_prediction).numpy()

    test_srcc, test_krcc, test_plcc, test_rmse = cal(all_prediction, all_authentic)

    metrics = torch.tensor([test_srcc, test_krcc, test_plcc, test_rmse], device=rank)
    dist.broadcast(metrics, src=0)
    test_srcc, test_krcc, test_plcc, test_rmse = metrics.cpu().numpy()

    if rank == 0:
        if best is not None:
            if test_srcc > best[0]:
                best[:4] = [test_srcc, test_krcc, test_plcc, test_rmse]
                torch.save(model.module.state_dict(), save_path)
            logger.info('test_srcc: {:.4f}, test_krcc: {:.4f}, test_plcc: {:.4f}, test_rmse: {:.4f}'
                        .format(*best))
        else:
            logger.info('val_srcc: {:.4f}, val_krcc: {:.4f}, val_plcc: {:.4f}, val_rmse: {:.4f}'
                        .format(test_srcc, test_krcc, test_plcc, test_rmse))

    return test_srcc, test_krcc, test_plcc, test_rmse


def train_one_epoch(model, criterion, train_loader, optimizer, epoch, rank,
                    lr_scheduler=None, args=None, logger=None):
    model.train()
    criterion.train()
    authentic = []
    prediction = []
    losses = []

    train_sampler = train_loader.sampler
    if hasattr(train_sampler, 'set_epoch'):
        train_sampler.set_epoch(epoch)

    for image, mos in tqdm(train_loader, disable=(rank != 0)):
        image = image.to(f'cuda:{rank}')
        mos = mos.to(f'cuda:{rank}')
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(image)
            authentic.append(mos.detach().cpu().numpy())
            prediction.append(pred.detach().cpu().numpy())
            loss = criterion(pred, mos)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
    train_srcc, train_krcc, train_plcc, train_rmse = cal(np.array(prediction), np.array(authentic))
    if rank == 0:
        logger.info(
            'train_epoch: {}, train_loss: {:.4f}, train_srcc: {:.4f}, train_krcc: {:.4f}, train_plcc: {:.4f}, train_rmse: {:.4f}'
            .format(epoch, sum(losses) / len(losses), train_srcc, train_krcc, train_plcc, train_rmse))

    return sum(losses) / len(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransformerXL training scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)

    # create log directory
    timestamp = time.strftime('_%Y%m%d_%H%M%S', time.localtime())
    args.output_dir = os.path.join(args.output_dir, os.path.basename(args.config_file)[:-3] + timestamp)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # whether to use distributed training
    if args.distributed:
        # os.environ['RANK'] = "0"
        # os.environ['WORLD_SIZE'] = "4"
        # os.environ['MASTER_ADDR'] = "127.0.0.1"
        # os.environ['MASTER_PORT'] = "3002"
        
        dist.init_process_group("nccl")
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        logging.info("=" * 100, args.rank, args.world_size)
        torch.distributed.barrier()
        misc.setup_for_distributed(args.rank == 0)
    else:
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    # Load Config and update args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        try:
            cfg.dump(save_cfg_path)
        except Exception:
            pass
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # load dictionary
    if "ascii" in args.dataset_name:
        word2id_dict = get_ascii_word2id_dict()
    else:
        word2id_dict = get_word2id_dict(args.vocab_path)

    # set global random seed
    seed = args.random_seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if 'spm_image_bpe_16384' in args.vocab_path:
        args.vocab_size = 16384
    else:
        args.vocab_size = len(word2id_dict)

    # build l3tc model
    model, criterion = build_model_main(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    # set logger
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="transformer_text_compression")
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info("args: " + str(args) + '\n')

    n_parameters_without_embed_fc = 0
    for name, val in model.named_parameters():
        if "rwkv" in args.model_name:
            if "head" in name or "emb" in name:
                continue
        elif args.model_name == "transformer":
            if "token_embed" in name or "pos_embed" in name or "generator" in name:
                continue
        elif "transformer_xl" in args.model_name:
            if "token_emd" in name:
                continue

        n_parameters_without_embed_fc += val.numel()
    logger.info('number of params without embed && fc:'+str(n_parameters_without_embed_fc))
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    # logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    # build TrainTestDataset
    dataset_name = getattr(args, "dataset_name", "default")
    if dataset_name == "enwik":
        dataset_train = EnWikTrainDataSet(args, args.train_file, word2id_dict)
        dataset_val = EnWikTestDataSet(args, args.test_file, word2id_dict)
    elif dataset_name == "enwik_ascii":
        dataset_train = EnWikASCIITrainDataSet(args, args.train_file, word2id_dict)
        dataset_val = EnWikASCIITestDataSet(args, args.test_file, word2id_dict)
    else:
        dataset_train = TransformerXLTrainDataSet(args, args.train_file, word2id_dict)
        dataset_val = TransformerXLTestDataSet(args, args.test_file, word2id_dict)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   num_workers=args.num_workers)

    eval_batch_size = 1024 if 'rwkv' in args.model_name else 1
    data_loader_val = DataLoader(dataset_val, batch_size=eval_batch_size, sampler=sampler_val,
                                 drop_last=False, num_workers=args.num_workers)

    # build optimizer and loss function
    if 'rwkv' in args.model_name:
        optimizer = model_without_ddp.configure_optimizers(args)
    else:
        optimizer = Adam(model_without_ddp.parameters(), lr=args.learning_rate)

    if args.scheduler[0] == "multi_epoch":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler[1], gamma=args.scheduler[2])
    elif args.scheduler[0] == "step_lr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler[1], gamma=args.scheduler[2])
    elif args.scheduler[0] == "exponential_lr":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler[1])
    else:
        lr_scheduler = None

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])           

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # load pretrained model
    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        # _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        _ignorekeywordlist = []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in misc.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))     
    
    if args.eval:
        test_stats = evaluate(model, data_loader_val)
        test_stats = evaluate(model, data_loader_val)
        sys.exit()

    # best result holder
    best_map_holder = BestMetricHolder(init_res=100.0, better='small', use_ema=False)

    # start training
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch,
                                      args.clip_max_norm, lr_scheduler=lr_scheduler if args.scheduler != "multi_epoch" else None, args=args, 
                                      logger=(logger if args.save_log else None) )
        
        if args.scheduler != "multi_epoch" and lr_scheduler is not None:
            lr_scheduler.step()

        if args.output_dir:
            # traverse current checkpoint
            all_exist_ckpts = glob.glob(os.path.join(args.output_dir, "*pth"))
            all_exist_ckpts = [ckptname for ckptname in all_exist_ckpts if os.path.basename(ckptname) not in ['checkpoint.pth', 'checkpoint_best.pth']]
            all_exist_ckpts = sorted(all_exist_ckpts)
            if len(all_exist_ckpts) >= 3:
                rm_cmd = "rm -f {}".format(all_exist_ckpts[0])
                os.system(rm_cmd)

            # checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint.pth')]
            checkpoint_paths = []
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    'epoch': epoch,
                    'args': args,
                }
                misc.save_on_master(weights, checkpoint_path)
        
        test_stats = evaluate(model, data_loader_val)
        avg_cross_entropy = test_stats['avg_ce']
        _isbest = best_map_holder.update(avg_cross_entropy, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
            misc.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats, indent=4) + "\n")
        
        if args.debug:
            if epoch >= 1:
                print("BREAK!"*5)
                break