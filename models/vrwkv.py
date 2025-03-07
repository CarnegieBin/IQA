# Copyright (c) Shanghai AI Lab. All rights reserved.
import torch
import torch.nn as nn
from typing import Sequence
import math
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from models.utils import DropPath
import os
import logging
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed
from mmcls.models.backbones.base_backbone import BaseBackbone

logger = logging.getLogger(__name__)

os.environ['PATH'] = "/data2/CarnegieBin_data/anaconda3/envs/kaniqa/bin:" + os.environ.get('PATH', '')
T_MAX = 4096
from torch.utils.cpp_extension import load

wkv_cuda = load(name="wkv", sources=["/data2/CarnegieBin_data/RWKVQA/models/cuda/wkv_op.cpp",
                                     "/data2/CarnegieBin_data/RWKVQA/models/cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=[
                    '-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3',
                    '-Xptxas -O3', f'-DT_MAX={T_MAX}'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1 / 4, patch_resolution=None):
    assert gamma <= 1 / 4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :,
                                                                         shift_pixel:W]
    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3),
                                                                         0:H - shift_pixel, :]
    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:,
                                                                             int(C * gamma * 3):int(C * gamma * 4),
                                                                             shift_pixel:H, :]
    output[:, int(C * gamma * 4):, ...] = input[:, int(C * gamma * 4):, ...]
    return output.flatten(2).transpose(1, 2)


def double_shift(input, shift_pixel=1, gamma=1 / 4, patch_resolution=None):
    assert gamma <= 1 / 2
    B, N, C = input.shape
    output = torch.zeros_like(input)
    output[:, shift_pixel:N, int(C * gamma * 2):int(C * gamma * 3)] = input[:, 0:N - shift_pixel,
                                                                      int(C * gamma * 2):int(C * gamma * 3)]
    output[:, 0:N - shift_pixel, int(C * gamma * 3):int(C * gamma * 4)] = input[:, shift_pixel:N,
                                                                          int(C * gamma * 3):int(C * gamma * 4)]
    return output


def single_shift(input, shift_pixel=1, *args):
    assert len(input.shape) == 3  # [B, N, C]
    return nn.ZeroPad2d((0, 0, shift_pixel, -shift_pixel))(input)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim: int,  p: float = -1.0,  eps: float = 1e-8, bias: bool = False):
        super(RMSNorm, self).__init__()
        assert p <= 1.0 and p >= 0.0 or p == -1.0, "Invalid partial ratio p (valid range: 0.0 <= p <= 1.0)"
        self.embed_dim = embed_dim
        self.p = p
        self.eps = eps
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(embed_dim))
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p < 0.0 or self.p > 1.0:
            norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
            d_x = self.embed_dim
        else:
            partial_size = int(self.embed_dim * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.embed_dim - partial_size], dim=-1)
            norm_x = torch.norm(partial_x, p=2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-0.5)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, p={self.p}, eps={self.eps}, bias={self.bias}"


class VRWKV_SpatialMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1 / 4, shift_pixel=1, init_mode='fancy',
                 key_norm=False, with_cp=False, norm=None):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd) if norm == 'LayerNorm' else RMSNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))  # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0

                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            sr, k, v = self.jit_func(x, patch_resolution)
            x = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
            if self.key_norm is not None:
                x = self.key_norm(x)
            x = sr * x
            x = self.output(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1 / 4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False, with_cp=False, norm=None):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz) if norm == 'LayerNorm' else RMSNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.shift_pixel > 0:
                xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            else:
                xk = x
                xr = x

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class Block(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1 / 4, shift_pixel=1, drop_path=0., hidden_rate=None,
                 init_mode='fancy', init_values=None, post_norm=False, key_norm=False,
                 with_cp=False, norm=None):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd) if norm == 'LayerNorm' else RMSNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) if norm == 'LayerNorm' else RMSNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd) if norm == 'LayerNorm' else RMSNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, init_mode,
                                    key_norm=key_norm, norm=norm)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, hidden_rate,
                                    init_mode, key_norm=key_norm, norm=norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


from typing import Optional, Sequence, Dict, List, Tuple
import torch
from torch import nn
from torch.nn import ModuleList


@BACKBONES.register_module()
class Net(BaseBackbone):
    def __init__(
            self,
            img_size: Tuple[int, int] = (512, 384),
            patch_size: int = 16,
            in_channels: int = 3,
            out_indices: Sequence[int] = (2, 4, 6, 8),
            drop_rate: float = 0.0,
            embed_dims: int = 768,
            depth: int = 9,
            drop_path_rate: float = 0.0,
            channel_gamma: float = 1 / 4,
            shift_pixel: int = 1,
            shift_mode: str = 'q_shift',
            init_mode: str = 'fancy',
            post_norm: bool = True,
            key_norm: bool = True,
            init_values: Optional[float] = 1e-6,
            hidden_rate: float = 8 / 3,
            final_norm: bool = True,
            interpolate_mode: str = 'bicubic',
            with_cp: bool = False,
            init_cfg: Optional[Dict] = None,
            norm: str = 'RMSNorm'
    ) -> None:
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )
        self.patch_resolution: Tuple[int, int] = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed: nn.Parameter = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # Process out_indices
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] += depth
            assert 0 <= out_indices[i] < depth, f'Invalid out_indices {out_indices[i]}'

        # Ensure unique and sorted indices
        assert len(out_indices) == len(set(out_indices)), "out_indices must be unique"
        self.out_indices = sorted(out_indices)
        self.index_map: Dict[int, int] = {index: idx for idx, index in enumerate(self.out_indices)}

        # CNN blocks with adaptive pooling to ensure consistent output size
        self.cnn_blocks = ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=3, padding=1),
                nn.AdaptiveAvgPool2d((1, 1))  # Ensure consistent output size
            ) for _ in self.out_indices
        ])

        # Stochastic depth
        dpr: List[float] = torch.linspace(0, drop_path_rate, depth).tolist()

        self.layers: ModuleList[Block] = ModuleList()
        for i in range(depth):
            self.layers.append(Block(
                n_embd=embed_dims,
                n_layer=depth,
                layer_id=i,
                channel_gamma=channel_gamma,
                shift_pixel=shift_pixel,
                shift_mode=shift_mode,
                hidden_rate=hidden_rate,
                drop_path=dpr[i],
                init_mode=init_mode,
                post_norm=post_norm,
                key_norm=key_norm,
                init_values=init_values,
                with_cp=with_cp,
                norm=norm
            ))

        # Final components
        self.final_norm = final_norm
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(embed_dims, 1, bias=True)

        if final_norm:
            norm_layer = nn.LayerNorm(embed_dims) if norm == 'LayerNorm' else RMSNorm(embed_dims)
            self.ln1: nn.Module = norm_layer

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x, patch_resolution = self.patch_embed(x)

        # Position embedding
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens
        )
        x = self.drop_after_pos(x)

        outs: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x = layer(x, patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.index_map:
                B, _, C = x.shape
                patch_token = x.view(B, *patch_resolution, C).permute(0, 3, 1, 2)
                outs.append(patch_token)

        return tuple(outs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        outs = []
        for feature, cnn_block in zip(features, self.cnn_blocks):
            out = cnn_block(feature)
            outs.append(out)

        # Feature fusion
        fused = torch.stack(outs, dim=0).sum(dim=0)
        pooled = self.pool(fused)
        return self.head(pooled.flatten(1))


if __name__ == '__main__':
    import shutil

    print("Ninja path:", shutil.which('ninja'))

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224) # 确保输入尺寸正确
    outputs = Net.forward(x)

    print("\nModel outputs:")
    for i, out in enumerate(outputs):
        print(f"Output {i + 1} shape: {out.shape}")  # 预期输出 [2, 256, 14, 14]

    print("\nOutput value range:")
    for out in outputs:
        print(f"Min: {out.min().item():.4f}, Max: {out.max().item():.4f}")
    # except Exception as e:
    #     print(f"Error occurred: {str(e)}")


