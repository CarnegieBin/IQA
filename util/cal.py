import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


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

    init = np.array([np.max(y), 0.5, np.mean(x), 0.5, 0.5])  # 更合理的初始参数示例
    popt, _ = curve_fit(func, x, y, p0=init, maxfev=100000)
    fitted = func(x, *popt)

    return None, None, fitted  # 仅返回拟合后的预测值

