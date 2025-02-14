import numpy as np
from torch.optim import AdamW
import torch
from scipy import stats, optimize


def build_optimizer_scheduler(model, args):
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args)
    return optimizer, scheduler


def build_scheduler(optimizer, args):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch - args.warmup_epoch
    )


def calculate_metrics(pred, mos):
    srcc, _ = stats.spearmanr(pred, mos)
    krcc, _ = stats.kendalltau(pred, mos)

    _, _, fitted_pred = logistic_5_fitting_no_constraint(pred, mos)
    plcc, _ = stats.pearsonr(fitted_pred, mos)
    rmse = np.sqrt(np.mean((fitted_pred - mos) ** 2))

    srcc = round(srcc, 4)
    krcc = round(krcc, 4)
    plcc = round(plcc, 4)
    rmse = round(rmse, 4)

    return srcc, krcc, plcc, rmse


def logistic_5_fitting_no_constraint(x, y):
    def func(x, b0, b1, b2, b3, b4):
        logistic_part = 0.5 - 1.0 / (1 + np.exp(b1 * (x - b2)))
        return b0 * logistic_part + b3 * x + b4

    # 改进点1：更合理的初始参数设置 ------------------------------
    # 基于数据特性动态初始化
    y_range = np.max(y) - np.min(y)
    x_mean = np.mean(x)

    # 初始参数建议：
    # b0: 控制逻辑部分幅值，初始化为 y 的范围
    # b1: 控制逻辑函数陡峭度，初始化为小值避免梯度爆炸
    # b2: 逻辑函数中点，初始化为 x 的均值
    # b3, b4: 线性部分参数，初始化为线性回归结果
    slope, intercept = np.polyfit(x, y, 1)  # 先用线性回归初始化
    init = [y_range * 0.5, 0.1, x_mean, slope, intercept]

    # 改进点2：数据标准化 ------------------------------
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)

    try:
        # 改进点3：增加边界约束和容差 ----------------------
        popt, _ = optimize.curve_fit(
            func,
            x_norm,
            y_norm,
            p0=init,
            maxfev=200000,  # 适当增大迭代次数
            bounds=(
                [-np.inf, -10, -np.inf, -np.inf, -np.inf],  # 下限
                [np.inf, 10, np.inf, np.inf, np.inf]  # 上限（限制b1范围）
            ),
            ftol=1e-6,  # 减小收敛容差
            xtol=1e-6
        )

        # 反标准化拟合结果
        fitted_norm = func(x_norm, *popt)
        fitted = fitted_norm * np.std(y) + np.mean(y)
    except Exception as e:
        print(f"Fitting failed: {e}")
        # 退回线性回归作为保底
        fitted = slope * x + intercept

    return None, None, fitted

