import os
import torch
import json
import time
import numpy as np
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from util.logger import setup_logger
from dataset.IQA_dataloader import get_dataloader
from util.utils import calculate_metrics as cal
from util.utils import build_optimizer_scheduler as bop
from models.vrwkv import Net
from torch.utils.data import DataLoader
from datetime import datetime


def get_args_parser():
    parser = argparse.ArgumentParser('IQA Training', add_help=False)

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=3e-4,
                        help="Initial learning rate (suggested: 1e-4 ~ 5e-4)")
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help="Weight decay coefficient (AdamW-specific, suggested: 0.01~0.1)")
    parser.add_argument('--beta1', type=float, default=0.9,
                        help="AdamW's beta1 parameter")
    parser.add_argument('--beta2', type=float, default=0.999,
                        help="AdamW's beta2 parameter")
    parser.add_argument('--eps', type=float, default=1e-8,
                        help="Numerical stability term (epsilon)")
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help="Gradient clipping threshold (0 to disable)")
    parser.add_argument('--warmup_epoch', type=int, default=5,
                        help="Number of epochs for learning rate warmup")
    # ===================================

    # Training parameters
    parser.add_argument('--output_dir', default='./ckpt', type=str,
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', default='./log', type=str,
                        help='Directory to save log file')
    parser.add_argument('--seed', default=1216, type=int,
                        help='Random seed')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data loading workers per process')
    parser.add_argument('--find_unused_params', action='store_true',
                        help='Find unused parameters in DDP')
    parser.add_argument('--amp', action='store_true',
                        help="Enable mixed precision training")
    parser.add_argument('--epoch', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 2, 3, 6, 7])
    # ===================================

    # Data parameters
    parser.add_argument('--dataset', default='RoIciCIQAD', type=str,
                        help='Dataset name for training')
    parser.add_argument('--patch_num', default=10, type=int,
                        help=' ')
    return parser


def evaluate(model, loader, logger, save_path=None, best_metrics=None):
    model.eval()
    authentic, prediction = [], []

    with torch.no_grad():
        for img, mos in tqdm(loader):
            img = img.cuda()
            pred = model(img).flatten()
            authentic.append(mos.cpu().numpy())
            prediction.append(pred.cpu().numpy())

    authentic = np.concatenate(authentic)
    prediction = np.concatenate(prediction)
    metrics = cal(prediction, authentic)

    if save_path and best_metrics:
        if metrics[0] > best_metrics[0]:
            torch.save(model.state_dict(), save_path)
            best_metrics[:] = metrics
            logger.info(f'New best model saved! Metrics: {metrics}')
    return metrics


def train_epoch(model, criterion, loader, optimizer, epoch, scaler, args, logger):
    model.train()
    losses = []

    for img, mos in tqdm(loader):
        img = img.cuda(non_blocking=True)
        mos = mos.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(img).flatten()
            loss = criterion(pred, mos)

        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        losses.append(loss.item())

    avg_loss = np.mean(losses)
    logger.info(f'Epoch {epoch} Train Loss: {avg_loss:.4f}')
    return avg_loss


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 初始化模型
    model = Net().cuda()
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    criterion = torch.nn.MSELoss().cuda()

    # 数据加载
    train_set, val_set, test_set = get_dataloader(args)
    gpu_num = len(args.gpu_ids)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size * gpu_num,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(val_set, batch_size=args.batch_size * gpu_num, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size * gpu_num, shuffle=False)

    # 优化器
    optimizer, scheduler = bop(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # 日志系统
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    log_path = os.path.join(args.log_dir, args.dataset, timestamp)
    Path(log_path).mkdir(parents=True, exist_ok=True)  # 自动创建嵌套目录

    # 初始化日志系统（日志文件保存在日志路径下）
    logger = setup_logger(
        output=os.path.join(log_path, 'log.txt'),  # 统一保存到带时间戳的目录
        name='IQA'
    )

    # 保存训练参数到同目录（包含完整配置的副本）
    args_path = os.path.join(log_path, 'runtime_args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)  # 添加排序保证可读性

    # 训练循环
    best_val = [0.0, 0.0, 0.0, 0.0]
    best_test = [0.0, 0.0, 0.0, 0.0]
    save_path = os.path.join(args.output_dir, args.dataset + '_' + 'best.pth')

    for epoch in range(args.epoch):
        train_epoch(model, criterion, train_loader, optimizer, epoch, scaler, args, logger)
        _ = evaluate(model, val_loader, logger, save_path, best_val)

        test_metrics = evaluate(model, test_loader, logger, save_path=None, best_metrics=best_test)
        logger.info(f'Final Test | SRCC: {test_metrics[0]:.4f} KRCC: {test_metrics[1]:.4f} '
                    f'PLCC: {test_metrics[2]:.4f} RMSE: {test_metrics[3]:.4f}')


