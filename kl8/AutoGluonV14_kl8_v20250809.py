#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KL‑8 probabilistic forecasting – 深度优化版本 AutoGluon 1.4.0 pipeline

本模块实现了针对KL‑8彩票数据集的端到端时间序列预测工作流程，
采用多种优化策略以提高预测准确性：

核心优化策略：
1. 智能特征工程：增加趋势、周期性、滞后特征
2. 多层次预测融合：位置级、全局级、统计级预测融合
3. 动态数字选择算法：基于多重评分机制
4. 高级预处理：异常值处理、数据平滑
5. 模型集成优化：自定义超参数和模型权重
6. 预测后处理：数字频率分析和约束优化

Author: Optimized by Claude
Updated: 2025‑08‑05
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.common import space

# ===== 日志配置 ================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===== 全局配置 ================================================================
# 预测参数和模型配置
# 调整预测分位点范围以避免 Chronos 系列模型警告。
# 使用 0.1–0.9 区间，避免使用 Chronos 未训练过的 0.05/0.95 分位数。
QUANTILE_LEVELS: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PREDICTION_LENGTH: int = 1  # 预测窗口长度
MAX_LAG: int = 30  # 用于创建滞后特征的最大历史窗口

# 使用 AutoGluon 1.4.0 新版本中的完整预设
# 在 v1.4.0 中，`best_quality` 预设包含了深度学习、树模型、统计模型
# 以及新引入的 PerStepTabular 等模型【794457681487353†L849-L863】。
# 因此无需手动指定 hyperparameters 即可使用最新模型。
PRESETS: str = "best_quality" # "fast_training"
MODEL_PATH: str = "autogluon_kl8_model_optimized"
NUM_VAL_WINDOWS: int = 5  # 使用多窗口交叉验证以提高泛化能力
TIME_LIMIT: Optional[int] = 36000  # 训练的最大时间（秒）

# 已知协变量列表：这些宏观特征在预测未来时可提前获取或通过外推估计。
# 在模型训练时，如果数据中包含下列列，则将其作为 known_covariates_names 传递给 AutoGluon。
KNOWN_COVARIATES: List[str] = [
    'sales_amount', 'sales_log', 'sales_pct_change',
    'pool_amount', 'pool_log', 'pool_pct_change', 'pool_to_sales'
]

# ==============================================================================
# 新增参数：高阶优化配置
#
# 这些参数用于改进后处理算法。您可以根据回测结果调整这些值。
# HIST_WINDOW: 计算历史频率和分布时使用的最近期数。
# WEIGHT_MEAN: 均值预测在评分中的权重。
# WEIGHT_QUANTILE: 分位数预测在评分中的权重。
# WEIGHT_RECENT_FREQ / WEIGHT_HIST_FREQ: 最近频率与全局频率的组合权重。
# HOT_QUANTILE / COLD_QUANTILE: 定义热号和冷号的分位数阈值。
HIST_WINDOW: int = 200
WEIGHT_MEAN: float = 3.0
WEIGHT_QUANTILE: float = 1.0
WEIGHT_RECENT_FREQ: float = 0.7
WEIGHT_HIST_FREQ: float = 0.3
HOT_QUANTILE: float = 0.9
COLD_QUANTILE: float = 0.1

# 是否使用遗传算法优化最终数字选择。若设为 True，则在评分后通过遗传算法搜索20个号码组合。
USE_GENETIC_ALGORITHM: bool = True

# ==========================================================================
# 新增优化参数：超参数调优和不确定度惩罚
#
# 为进一步提升模型预测准确率，我们引入了自定义的超参数搜索空间和
# 自动超参数调优机制。文档指出，用户可以定义模型的参数搜索范围并传入
# `hyperparameter_tune_kwargs`，AutoGluon 会根据验证结果挑选最佳配置【115439681625210†L1314-L1333】。
# 该过程会训练多个不同配置的模型并保留性能最佳者，但会增加训练时间【115439681625210†L1334-L1364】。
#
# HPO_NUM_TRIALS: 每个模型的随机搜索试验次数。
# HYPERPARAMETERS_TUNING: 定义深度学习模型的搜索空间。
# HYPERPARAMETER_TUNE_KWARGS: 控制调优过程的总试验次数及搜索策略。
#
# 我们还新增 WEIGHT_UNCERTAINTY，用于根据预测分位数间距惩罚高不确定度的数字，
# 通过对 P90-P10 的宽度进行标准化并从得分中减去一定比例，实现偏好更确定的预测。

HPO_NUM_TRIALS: int = 12  # 每个模型的随机搜索试验次数（可根据计算资源调整）

# 自定义超参数搜索空间：仅针对部分深度模型设置搜索范围
HYPERPARAMETERS_TUNING: Optional[Dict] = {
    "DeepAR": {
        # 隐藏层大小 20–80；更多层可以捕捉复杂模式
        "hidden_size": space.Int(20, 80),
        # 隐藏层数 1–3
        "num_layers": space.Int(1, 3),
        # dropout 防止过拟合
        "dropout_rate": space.Categorical(0.1, 0.2, 0.3),
    },
    "TemporalFusionTransformer": {
        "hidden_size": space.Int(16, 64),
        "dropout_rate": space.Categorical(0.1, 0.2, 0.3),
    },
    "PatchTST": {
        # patch 大小影响局部时间模式捕捉能力
        "patch_size": space.Categorical(4, 8, 16),
        "hidden_size": space.Int(16, 64),
    },
    # 引入 Chronos-Bolt 预训练模型，提供更快的推理速度
    "ChronosModel": {
        # 在 Bolt-mini 与 Bolt-small 之间随机选择，提高模型多样性
        "model_path": space.Categorical(
            "autogluon/chronos-bolt-mini",
            "autogluon/chronos-bolt-small",
        ),
    },
}

# 调参配置：控制总试验次数以及搜索策略
HYPERPARAMETER_TUNE_KWARGS: Optional[Dict[str, any]] = {
    "num_trials": HPO_NUM_TRIALS,
    "scheduler": "local",
    "searcher": "random",
}

# 不确定度惩罚权重：根据预测分位差异（例如 90% 分位与 10% 分位差）降低对应数字分数。
# 值越大，越倾向选择分位区间更窄（置信度更高）的数字。
WEIGHT_UNCERTAINTY: float = 0.5


# 在 1.4.0 版本中不再单独微调模型，而是采用预设中的默认参数集。
# 因此不需要传递 custom hyperparameters；只需依赖 `presets`。
HYPERPARAMETERS: Optional[Dict] = None

# ===== 辅助函数：根据模型得分计算权重 =====
def compute_model_weights(predictor: TimeSeriesPredictor, tdf: TimeSeriesDataFrame) -> Dict[str, float]:
    """
    根据模型在验证集上的得分计算权重，权重用于组合多模型预测。
    AutoGluon 的 leaderboard 返回每个模型的 score_val（越大越好或越小越好视评估指标而定）。
    当使用 WQL（加权分位损失）时，score_val 通常是负值且数值越大代表性能越好，因此将其乘以 -1 以获得正值。
    如果无法获取 leaderboard，则返回等权重。

    Parameters
    ----------
    predictor : TimeSeriesPredictor
        已训练好的时序预测器。
    tdf : TimeSeriesDataFrame
        用于评估各模型性能的数据集。

    Returns
    -------
    Dict[str, float]
        每个模型对应的权重，权重之和为 1。
    """
    try:
        lb_df = predictor.leaderboard(tdf, silent=True)
        if not isinstance(lb_df, pd.DataFrame):
            raise ValueError("leaderboard did not return a DataFrame")
        if 'model' in lb_df.columns and 'score_val' in lb_df.columns:
            perf = lb_df.set_index('model')['score_val'].to_dict()
            scores: Dict[str, float] = {}
            for m, v in perf.items():
                val = -v if v < 0 else v
                scores[m] = max(val, 0)
            total = sum(scores.get(m, 0) for m in predictor.model_names())
            if total == 0:
                n = len(predictor.model_names())
                return {m: 1 / n for m in predictor.model_names()}
            return {m: scores.get(m, 0) / total for m in predictor.model_names()}
    except Exception as e:
        logger.warning(f"无法计算模型权重，使用等权重: {e}")
    n = len(predictor.model_names())
    return {m: 1 / n for m in predictor.model_names()}

# ===== 辅助函数：根据模型得分计算权重 =====
def compute_model_weights(predictor: TimeSeriesPredictor, tdf: TimeSeriesDataFrame) -> Dict[str, float]:
    """
    根据模型在验证集上的得分计算权重，权重用于组合多模型预测。
    AutoGluon 的 leaderboard 返回每个模型的 score_val（越大越好或越小越好视评估指标而定）。
    当使用 WQL（加权分位损失）时，score_val 通常是负值且数值越大代表性能越好，因此将其乘以 -1 以获得正值。
    如果无法获取 leaderboard，则返回等权重。

    Parameters
    ----------
    predictor : TimeSeriesPredictor
        已训练好的时序预测器。
    tdf : TimeSeriesDataFrame
        用于估计 leaderboard 的数据集。

    Returns
    -------
    Dict[str, float]
        每个模型对应的权重，权重之和为 1。
    """
    try:
        # 使用传入的 tdf 评估各模型性能
        lb_df = predictor.leaderboard(tdf, silent=True)
        if not isinstance(lb_df, pd.DataFrame):
            raise ValueError("leaderboard did not return a DataFrame")
        # 仅在存在 'model' 和 'score_val' 列时进行加权计算
        if 'model' in lb_df.columns and 'score_val' in lb_df.columns:
            perf = lb_df.set_index('model')['score_val'].to_dict()
            # 将分数转换为正值，值越大表示模型越好
            scores: Dict[str, float] = {}
            for m, v in perf.items():
                # 如果评估指标为 WQL，则 score_val 为负，使用 -v 使其为正；否则取绝对值
                val = -v if v < 0 else v
                scores[m] = max(val, 0)
            total = sum(scores.get(m, 0) for m in predictor.model_names())
            if total == 0:
                # 如果所有得分都为 0，则返回等权重
                n = len(predictor.model_names())
                return {m: 1 / n for m in predictor.model_names()}
            # 归一化得到权重
            return {m: scores.get(m, 0) / total for m in predictor.model_names()}
    except Exception as e:
        logger.warning(f"无法计算模型权重，使用等权重: {e}")
    # 默认等权重
    n = len(predictor.model_names())
    return {m: 1 / n for m in predictor.model_names()}

# ===== 数据加载和预处理 =========================================================

def load_data(path: str) -> pd.DataFrame:
    """加载KL-8原始数据并进行基础清理"""
    df = pd.read_csv(path, encoding='utf-8')

    if '开奖日期' in df.columns:
        df = df.rename(columns={'开奖日期': 'timestamp'})

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ignore_index=True)

    # 基础数据质量检查
    logger.info(f"数据加载完成：{len(df)} 条记录，时间范围：{df['timestamp'].min()} 到 {df['timestamp'].max()}")

    return df


def extract_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取高级时间特征和统计特征"""
    df = df.copy()

    # 基础时间特征
    df['day_of_week'] = df['timestamp'].dt.dayofweek.astype('int8')
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype('int16')
    df['month'] = df['timestamp'].dt.month.astype('int8')
    df['day_of_month'] = df['timestamp'].dt.day.astype('int8')
    df['quarter'] = df['timestamp'].dt.quarter.astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')

    # 高级周期特征
    df['day_of_year'] = df['timestamp'].dt.dayofyear.astype('int16')
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # 数字分布统计特征
    num_cols = [f'开奖号_{i}' for i in range(1, 21)]
    if all(col in df.columns for col in num_cols):
        numbers_array = df[num_cols].values

        # 统计特征
        df['mean_number'] = np.mean(numbers_array, axis=1)
        df['std_number'] = np.std(numbers_array, axis=1)
        df['median_number'] = np.median(numbers_array, axis=1)
        df['range_number'] = np.ptp(numbers_array, axis=1)
        df['sum_number'] = np.sum(numbers_array, axis=1)

        # 分布特征
        df['skewness'] = stats.skew(numbers_array, axis=1)
        df['kurtosis'] = stats.kurtosis(numbers_array, axis=1)

        # 区间分布特征
        df['low_range_count'] = np.sum((numbers_array >= 1) & (numbers_array <= 20), axis=1)
        df['mid_range_count'] = np.sum((numbers_array >= 21) & (numbers_array <= 60), axis=1)
        df['high_range_count'] = np.sum((numbers_array >= 61) & (numbers_array <= 80), axis=1)

        # 奇偶性特征
        df['odd_count'] = np.sum(numbers_array % 2 == 1, axis=1)
        df['even_count'] = np.sum(numbers_array % 2 == 0, axis=1)

        # 连续性特征
        sorted_numbers = np.sort(numbers_array, axis=1)
        consecutive_diffs = np.diff(sorted_numbers, axis=1)
        df['consecutive_count'] = np.sum(consecutive_diffs == 1, axis=1)
        df['max_gap'] = np.max(consecutive_diffs, axis=1)
        df['min_gap'] = np.min(consecutive_diffs, axis=1)

        # === 自定义高级模式特征 ===
        # 1. 素数统计特征：计算每期中素数的数量和素数之和
        primes = np.array([
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
            43, 47, 53, 59, 61, 67, 71, 73, 79
        ], dtype=int)
        is_prime = np.isin(numbers_array, primes)
        df['prime_count'] = is_prime.sum(axis=1)
        prime_values = numbers_array * is_prime
        df['prime_sum'] = prime_values.sum(axis=1)

        # 2. 极值特征：最大的5个数之和与最小的5个数之和
        df['sum_top5'] = np.sum(sorted_numbers[:, -5:], axis=1)
        df['sum_bottom5'] = np.sum(sorted_numbers[:, :5], axis=1)

        # 3. 聚类特征：使用KMeans对号码模式进行聚类
        try:
            scaler = StandardScaler()
            numbers_scaled = scaler.fit_transform(numbers_array)
            km = KMeans(n_clusters=5, random_state=42)
            cluster_labels = km.fit_predict(numbers_scaled)
            df['num_cluster'] = cluster_labels.astype('int8')
        except Exception:
            # 如果聚类失败，填充0作为默认类别
            df['num_cluster'] = 0

        # ========== 时序差分特征 ==========
        # 为捕捉连续期次之间统计特征的变化趋势，计算相关指标的差分。差分能反映出上期与本期之间的升降变化，
        # 对模型学习趋势和波动信息有帮助。
        diff_cols = {
            'mean_diff': 'mean_number',
            'sum_diff': 'sum_number',
            'range_diff': 'range_number',
            'odd_diff': 'odd_count',
            'even_diff': 'even_count',
        }
        for new_col, base_col in diff_cols.items():
            if base_col in df.columns:
                df[new_col] = df[base_col].diff().fillna(0)

    # ==================== 宏观销售额与奖池特征 ====================
    # 部分数据集中包含《本期销售金额》和《选十玩法奖池金额》列，通常带有千位分隔符（逗号），需要转为数值。
    # 这些特征在开奖前即可获得，可作为已知协变量传递给 AutoGluon。
    # 销售额特征
    if '本期销售金额' in df.columns:
        # 将逗号（中英文）移除后转换为浮点型
        sales_raw = df['本期销售金额'].astype(str).str.replace(',', '', regex=False).str.replace('，', '', regex=False)
        df['sales_amount'] = pd.to_numeric(sales_raw, errors='coerce')
    if 'sales_amount' in df.columns:
        # 对销售额取对数，加入环比变化率
        df['sales_log'] = np.log1p(df['sales_amount'])
        df['sales_pct_change'] = df['sales_amount'].pct_change().fillna(0)

    # 奖池特征
    if '选十玩法奖池金额' in df.columns:
        pool_raw = df['选十玩法奖池金额'].astype(str).str.replace(',', '', regex=False).str.replace('，', '', regex=False)
        df['pool_amount'] = pd.to_numeric(pool_raw, errors='coerce')
    if 'pool_amount' in df.columns:
        df['pool_log'] = np.log1p(df['pool_amount'].fillna(0))
        df['pool_pct_change'] = df['pool_amount'].pct_change().fillna(0)
        # 奖池与销售额比率，避免除以零
        if 'sales_amount' in df.columns:
            df['pool_to_sales'] = df['pool_amount'] / df['sales_amount'].replace(0, np.nan)
        else:
            df['pool_to_sales'] = np.nan

    return df


def detect_and_handle_anomalies(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """检测和处理异常值"""
    df = df.copy()

    for col in num_cols:
        if col in df.columns:
            # 使用IQR方法检测异常值
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = max(1, Q1 - 1.5 * IQR)  # 确保不低于1
            upper_bound = min(80, Q3 + 1.5 * IQR)  # 确保不超过80

            # 异常值处理：用边界值替换
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def smooth_numeric_series(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """对数字列应用滚动均值平滑，降低噪声影响"""
    df = df.copy()
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].rolling(window=3, min_periods=1).mean()
    return df


def create_lag_features(df: pd.DataFrame, num_cols: List[str], lags: List[int]) -> pd.DataFrame:
    """创建滞后特征"""
    df = df.copy()

    for col in num_cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

            # 滚动统计特征
            for window in [3, 7, 14]:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window).max()
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window).min()

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """深度优化的数据预处理"""
    # 提取高级特征
    df = extract_advanced_features(df)

    # 数字列
    num_cols = [f'开奖号_{i}' for i in range(1, 21)]

    # 异常值处理
    df = detect_and_handle_anomalies(df, num_cols)

    # 滚动平滑，缓解短期剧烈波动
    df = smooth_numeric_series(df, num_cols)

    # 创建滞后特征
    lags = [1, 2, 3, 7, 14, 21]
    # 在数字列之外，为销售额、奖池和差分等特征生成滞后值，以捕捉这些宏观特征的变化趋势
    extra_lag_cols: List[str] = []
    for col in [
        'sales_amount', 'sales_log', 'sales_pct_change',
        'pool_amount', 'pool_log', 'pool_pct_change', 'pool_to_sales',
        'mean_diff', 'sum_diff', 'range_diff', 'odd_diff', 'even_diff'
    ]:
        if col in df.columns:
            extra_lag_cols.append(col)
    df = create_lag_features(df, num_cols + extra_lag_cols, lags)

    # 特征列表
    covariate_cols = [
        # 时间特征
        'day_of_week', 'week_of_year', 'month', 'day_of_month', 'quarter', 'is_weekend',
        'day_of_year', 'week_sin', 'week_cos', 'month_sin', 'month_cos',
        # 宏观销售额及奖池特征
        'sales_amount', 'sales_log', 'sales_pct_change',
        'pool_amount', 'pool_log', 'pool_pct_change', 'pool_to_sales',
        # 数字统计特征
        'mean_number', 'std_number', 'median_number', 'range_number', 'sum_number',
        'skewness', 'kurtosis', 'low_range_count', 'mid_range_count', 'high_range_count',
        'odd_count', 'even_count', 'consecutive_count', 'max_gap', 'min_gap',
        # 新增的模式特征
        'prime_count', 'prime_sum', 'sum_top5', 'sum_bottom5', 'num_cluster',
        # 时序差分特征
        'mean_diff', 'sum_diff', 'range_diff', 'odd_diff', 'even_diff'
    ]
    covariate_cols = [col for col in covariate_cols if col in df.columns]

    # 添加所有 lag/roll 特征
    lag_cols = [col for col in df.columns if '_lag_' in col or '_roll_' in col]
    covariate_cols.extend(lag_cols)

    # 转换为长格式
    long_vals = df.melt(
        id_vars=['timestamp'] + covariate_cols,
        value_vars=num_cols,
        var_name='pos',
        value_name='target'
    ).dropna(subset=['target'])

    # 构建类别标识符
    long_vals['item_id'] = long_vals['pos'].str.replace('开奖号_', 'pos_', regex=False)
    long_vals['target'] = long_vals['target'].astype(int)

    # 排序并创建行索引
    long_vals = long_vals.sort_values(['item_id', 'timestamp'], ignore_index=True)
    long_vals['row_idx'] = long_vals.groupby('item_id').cumcount()

    # 删除初始行以确保足够的滞后历史
    out = long_vals[long_vals['row_idx'] >= MAX_LAG].drop(columns=['pos', 'row_idx'])

    # 类型优化
    out['item_id'] = out['item_id'].astype('category')

    # 特征选择和缩放
    numeric_features = out.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['target']]

    # —— 新增清理步骤 ——
    # 把 ±inf 转成 NaN，避免后续 scaler 报错
    out = out.replace([np.inf, -np.inf], np.nan)

    # 填充缺失值（用中位数）
    for col in numeric_features:
        if out[col].isnull().sum() > 0:
            out[col] = out[col].fillna(out[col].median())

    # 全局 z-score 异常值检测与分组滚动平滑
    if numeric_features:
        z_scores = np.abs(stats.zscore(out[numeric_features], nan_policy='omit'))
        out[numeric_features] = np.where(z_scores > 3, np.nan, out[numeric_features])
        for col in numeric_features:
            if out[col].isnull().sum() > 0:
                out[col] = out[col].fillna(out[col].median())
        out = out.sort_values(['item_id', 'timestamp'])
        for col in numeric_features:
            out[col] = (
                out.groupby('item_id')[col]
                   .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            )

    logger.info(f"预处理完成：{len(out)} 条记录，{len(numeric_features)} 个特征")

    return out.reset_index(drop=True)



# ===== 模型训练 ================================================================

def train_model(train_df: pd.DataFrame) -> TimeSeriesPredictor:
    # Convert to TimeSeriesDataFrame
    tdf = TimeSeriesDataFrame.from_data_frame(
        train_df, id_column='item_id', timestamp_column='timestamp'
    ).fill_missing_values(method="auto")

    # Determine which known covariates are present
    present_known_covs = [c for c in KNOWN_COVARIATES if c in train_df.columns]

    # Instantiate predictor **with** known_covariates_names
    predictor = TimeSeriesPredictor(
        path=MODEL_PATH,
        prediction_length=PREDICTION_LENGTH,
        freq='D',
        target='target',
        eval_metric='WQL',
        quantile_levels=QUANTILE_LEVELS,
        known_covariates_names=present_known_covs or None  # pass here
    )

    # 构建训练参数
    fit_kwargs = {
        'train_data': tdf,
        'presets': PRESETS,
        'num_val_windows': NUM_VAL_WINDOWS,
        'refit_every_n_windows': 1,
        'time_limit': TIME_LIMIT,
        # 使用模型集成
        'enable_ensemble': True,
    }

    # 如果提供了超参数搜索空间，则配置调参参数
    if HYPERPARAMETERS_TUNING:
        fit_kwargs['hyperparameters'] = HYPERPARAMETERS_TUNING
        # 调参配置；如果指定 None 则使用默认策略
        fit_kwargs['hyperparameter_tune_kwargs'] = HYPERPARAMETER_TUNE_KWARGS
        # 当进行超参搜索时，可保持 ensemble 启用以集成不同超参版本的模型

    logger.info("开始模型训练…")
    predictor.fit(**fit_kwargs)
    logger.info("模型训练完成")

    return predictor



# ===== 智能数字选择算法 =========================================================

def analyze_historical_patterns(df: pd.DataFrame) -> Dict[str, any]:
    """分析历史数字模式"""
    # 使用动态历史窗口和分位数阈值分析历史数据
    df_sorted = df.sort_values('timestamp')
    num_cols = [f'开奖号_{i}' for i in range(1, 21)]
    # 全局频率
    all_numbers = df_sorted[num_cols].values.flatten()
    number_freq = pd.Series(all_numbers).value_counts().sort_index()
    # 最近窗口频率
    recent_df = df_sorted.tail(HIST_WINDOW)
    recent_numbers = recent_df[num_cols].values.flatten()
    recent_freq = pd.Series(recent_numbers).value_counts().sort_index()
    # 热号冷号阈值采用频率分位数
    hot_threshold = number_freq.quantile(HOT_QUANTILE)
    cold_threshold = number_freq.quantile(COLD_QUANTILE)
    hot_numbers = number_freq[number_freq >= hot_threshold].index.tolist()
    cold_numbers = number_freq[number_freq <= cold_threshold].index.tolist()
    # 计算低/中/高段以及奇数的历史分布，返回平均值
    def compute_distribution_stats(dataframe: pd.DataFrame) -> Dict[str, float]:
        low_counts, mid_counts, high_counts, odd_counts = [], [], [], []
        for nums in dataframe[num_cols].values:
            low_counts.append(sum(1 for n in nums if 1 <= n <= 20))
            mid_counts.append(sum(1 for n in nums if 21 <= n <= 60))
            high_counts.append(sum(1 for n in nums if 61 <= n <= 80))
            odd_counts.append(sum(1 for n in nums if n % 2 == 1))
        return {
            'low': float(np.mean(low_counts)),
            'mid': float(np.mean(mid_counts)),
            'high': float(np.mean(high_counts)),
            'odd': float(np.mean(odd_counts))
        }
    dist_stats_global = compute_distribution_stats(df_sorted)
    dist_stats_recent = compute_distribution_stats(recent_df)
    # 合并两者，最近窗口占70%，全局占30%
    dist_stats = {k: dist_stats_recent[k] * 0.7 + dist_stats_global[k] * 0.3 for k in dist_stats_global}
    return {
        'number_freq': number_freq,
        'recent_freq': recent_freq,
        'hot_numbers': hot_numbers,
        'cold_numbers': cold_numbers,
        'dist_stats': dist_stats
    }


def postprocess_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """对模型预测结果进行平滑和截断，增强后处理稳定性"""
    df = pred_df.copy()
    cols = ['mean'] + [str(q) for q in QUANTILE_LEVELS if str(q) in df.columns]
    for col in cols:
        df[col] = df[col].rolling(window=3, min_periods=1).mean().clip(1, 80)
    return df


def compute_advanced_final_numbers(
        predictions: pd.DataFrame,
        historical_patterns: Dict[str, any],
        original_df: Optional[pd.DataFrame] = None
) -> List[int]:
    """根据预测结果和历史模式计算最终的 20 个号码

    主要优化点：
        * 使用灵活的权重对均值和各分位数进行评分。
        * 根据最近和全局频率的加权和进行评分。
        * 热号和冷号加分根据热/冷阈值动态确定。
        * 调整候选集合大小根据预测数量分布、区间平均值等。
    """
    # 初始化分数
    number_scores: Dict[int, float] = {n: 0.0 for n in range(1, 81)}
    # 1. 预测评分：均值和分位数
    pred_columns = ['mean'] + [str(q) for q in QUANTILE_LEVELS]
    for col in pred_columns:
        if col not in predictions.columns:
            continue
        # 赋予不同权重
        weight = WEIGHT_MEAN if col == 'mean' else WEIGHT_QUANTILE
        for val in predictions[col]:
            num = int(np.clip(round(float(val)), 1, 80))
            number_scores[num] += weight
    # —— 不确定度惩罚 ——
    # 使用分位区间宽度作为不确定度度量。宽度越大，预测越不稳定，扣减更多分数。
    # 仅在存在足够分位列且启用惩罚时计算。
    if WEIGHT_UNCERTAINTY > 0:
        # 确定最低和最高分位列名
        try:
            min_q = str(min(QUANTILE_LEVELS))
            max_q = str(max(QUANTILE_LEVELS))
            if min_q in predictions.columns and max_q in predictions.columns:
                spreads = predictions[max_q] - predictions[min_q]
                # 防止除以零
                max_spread = float(spreads.max()) if len(spreads) > 0 else 1.0
                if max_spread == 0:
                    max_spread = 1.0
                for idx, row in predictions.iterrows():
                    pred_num = int(np.clip(round(float(row['mean'])), 1, 80))
                    # 正则化后的不确定度
                    unc = float(spreads.iloc[idx]) / max_spread
                    # 从分数中扣减不确定度权重
                    number_scores[pred_num] -= WEIGHT_UNCERTAINTY * unc
        except Exception:
            pass
    # 分数标准化：确保所有分数非负并归一化
    # 负分数裁剪为 0
    for num in number_scores:
        if number_scores[num] < 0:
            number_scores[num] = 0.0
    max_pred = max(number_scores.values()) if number_scores else 1.0
    if max_pred == 0:
        max_pred = 1.0
    for num in number_scores:
        number_scores[num] /= max_pred
    # 2. 历史频率评分
    number_freq = historical_patterns['number_freq']
    recent_freq = historical_patterns['recent_freq']
    # 对不同来源使用配置的权重
    max_hist = number_freq.max() if len(number_freq) > 0 else 1.0
    max_recent = recent_freq.max() if len(recent_freq) > 0 else 1.0
    for num in number_scores:
        hist_score = (number_freq.get(num, 0) / max_hist) * WEIGHT_HIST_FREQ
        recent_score = (recent_freq.get(num, 0) / max_recent) * WEIGHT_RECENT_FREQ
        number_scores[num] += hist_score + recent_score
    # 3. 冷热号加分
    hot_numbers = set(historical_patterns['hot_numbers'])
    cold_numbers = set(historical_patterns['cold_numbers'])
    for num in number_scores:
        if num in hot_numbers:
            number_scores[num] += 0.2  # 热号加分
        elif num in cold_numbers:
            number_scores[num] += 0.1  # 冷号少量加分
    # 4. 按分数排序选择候选数字
    sorted_scores = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
    # 根据分布平均数确定候选数目：每段多取一些候选，确保足够选择
    # 这里选取排名前40的号码作为候选，以便更大的搜索空间
    candidate_numbers = [num for num, _ in sorted_scores[:40]]
    # 根据配置决定使用哪种优化策略
    if USE_GENETIC_ALGORITHM:
        # 使用遗传算法优化选择
        final_numbers = genetic_algorithm_selection(candidate_numbers, number_scores, historical_patterns)
    else:
        # 使用规则型分布约束优化
        final_numbers = optimize_number_selection(candidate_numbers, historical_patterns)
    return sorted(final_numbers)


def optimize_number_selection(candidates: List[int], historical_patterns: Dict[str, any]) -> List[int]:
    """根据历史分布约束从候选数字中选出20个号码"""
    # 目标分布
    dist_stats = historical_patterns.get('dist_stats', {
        'low': 5.0, 'mid': 10.0, 'high': 5.0, 'odd': 10.0
    })
    target_low = max(0, min(20, int(round(dist_stats['low']))))
    target_mid = max(0, min(20, int(round(dist_stats['mid']))))
    target_high = max(0, min(20, int(round(dist_stats['high']))))
    # 调整使三者和为20
    total = target_low + target_mid + target_high
    if total != 20:
        target_mid += (20 - total)
        target_mid = max(0, target_mid)
    # 分配候选
    low_c = [n for n in candidates if 1 <= n <= 20]
    mid_c = [n for n in candidates if 21 <= n <= 60]
    high_c = [n for n in candidates if 61 <= n <= 80]
    selected: List[int] = []
    selected.extend(low_c[:target_low])
    selected.extend(mid_c[:target_mid])
    selected.extend(high_c[:target_high])
    # 不足则补充
    if len(selected) < 20:
        remaining = [n for n in candidates if n not in selected]
        need = 20 - len(selected)
        selected.extend(remaining[:need])
    if len(selected) < 20:
        remaining_all = [n for n in range(1, 81) if n not in selected]
        need = 20 - len(selected)
        selected.extend(remaining_all[:need])
    # 奇偶性调整
    target_odd = int(round(dist_stats['odd']))
    current_odd = sum(1 for n in selected if n % 2 == 1)
    if current_odd > target_odd + 2:
        # 用偶数替换多余的奇数
        for i, n in enumerate(selected):
            if n % 2 == 1:
                replacement = next((m for m in candidates if m % 2 == 0 and m not in selected), None)
                if replacement is None:
                    replacement = next((m for m in range(1, 81) if m % 2 == 0 and m not in selected), None)
                if replacement is not None:
                    selected[i] = replacement
                    current_odd -= 1
                if current_odd <= target_odd:
                    break
    elif current_odd < target_odd - 2:
        # 用奇数替换多余的偶数
        for i, n in enumerate(selected):
            if n % 2 == 0:
                replacement = next((m for m in candidates if m % 2 == 1 and m not in selected), None)
                if replacement is None:
                    replacement = next((m for m in range(1, 81) if m % 2 == 1 and m not in selected), None)
                if replacement is not None:
                    selected[i] = replacement
                    current_odd += 1
                if current_odd >= target_odd:
                    break
    return selected[:20]


# ===== 遗传算法优化数字选择 ====================================================
def genetic_algorithm_selection(candidates: List[int], number_scores: Dict[int, float],
                               historical_patterns: Dict[str, any],
                               population_size: int = 60, generations: int = 40,
                               crossover_rate: float = 0.7, mutation_rate: float = 0.3) -> List[int]:
    """
    使用简单遗传算法从候选数字中选择20个号码，适应度函数考虑预测评分和分布约束。

    Parameters
    ----------
    candidates : List[int]
        候选数字列表（例如评分前40名）。
    number_scores : Dict[int, float]
        每个数字的评分，用于适应度计算。
    historical_patterns : Dict[str, any]
        历史模式分析结果，包含目标分布。
    population_size : int
        种群规模。
    generations : int
        迭代代数。
    crossover_rate : float
        交叉概率。
    mutation_rate : float
        变异概率。

    Returns
    -------
    List[int]
        选择的20个号码。
    """
    import random
    # 目标分布
    dist_stats = historical_patterns.get('dist_stats', {'low': 5.0, 'mid': 10.0, 'high': 5.0, 'odd': 10.0})
    target_low = int(round(dist_stats['low']))
    target_mid = int(round(dist_stats['mid']))
    target_high = int(round(dist_stats['high']))
    target_odd = int(round(dist_stats['odd']))
    # 适应度函数：总评分减去分布偏差惩罚
    def fitness(individual: List[int]) -> float:
        score_sum = sum(number_scores.get(n, 0) for n in individual)
        # 分布
        low = sum(1 for n in individual if 1 <= n <= 20)
        mid = sum(1 for n in individual if 21 <= n <= 60)
        high = sum(1 for n in individual if 61 <= n <= 80)
        odd = sum(1 for n in individual if n % 2 == 1)
        # 计算偏差
        penalty = (abs(low - target_low) + abs(mid - target_mid) + abs(high - target_high)) * 0.05
        penalty += abs(odd - target_odd) * 0.03
        return score_sum - penalty
    # 初始种群：随机从候选中选取20个
    def random_individual():
        return sorted(random.sample(candidates, 20))
    population = [random_individual() for _ in range(population_size)]
    # 运行迭代
    for _ in range(generations):
        # 计算适应度
        fitness_values = [fitness(ind) for ind in population]
        # 选择：保留前30%精英
        elite_num = max(1, int(population_size * 0.3))
        elite_indices = sorted(range(population_size), key=lambda i: fitness_values[i], reverse=True)[:elite_num]
        elites = [population[i] for i in elite_indices]
        # 生成新种群
        new_population = elites.copy()
        while len(new_population) < population_size:
            if random.random() < crossover_rate and len(new_population) + 1 < population_size:
                # 交叉
                parents = random.sample(elites, 2)
                cut = random.randint(1, 19)
                child1 = parents[0][:cut] + [n for n in parents[1] if n not in parents[0][:cut]]
                child2 = parents[1][:cut] + [n for n in parents[0] if n not in parents[1][:cut]]
                # 修正长度
                child1 = sorted(child1[:20])
                child2 = sorted(child2[:20])
                new_population.extend([child1, child2])
            else:
                # 直接复制一个精英
                new_population.append(random.choice(elites))
        # 变异
        for i in range(elite_num, population_size):
            if random.random() < mutation_rate:
                ind = new_population[i]
                # 随机替换几个数字
                replace_count = random.randint(1, 3)
                to_replace = random.sample(range(20), replace_count)
                available = [n for n in candidates if n not in ind]
                for idx in to_replace:
                    if available:
                        ind[idx] = random.choice(available)
                        available.remove(ind[idx])
                ind.sort()
        population = new_population[:population_size]
    # 返回适应度最高的个体
    fitness_values = [fitness(ind) for ind in population]
    best_individual = population[int(np.argmax(fitness_values))]
    return best_individual


# ===== 预测导出 ================================================================

def export_next_period_forecasts(
    predictor: TimeSeriesPredictor,
    full_df: pd.DataFrame,
    original_df: pd.DataFrame,
    out_dir: str
):
    """
    生成并导出下一期的概率预测

    Parameters
    ----------
    predictor : TimeSeriesPredictor
        已训练好的时序预测器
    full_df : pd.DataFrame
        包含历史和未来已知协变量在内的长表格数据，列必须至少有 ['item_id','timestamp'] + KNOWN_COVARIATES
    original_df : pd.DataFrame
        原始历史数据，用于分析历史规律
    out_dir : str
        预测文件输出目录
    """
    # 分析历史模式（可选，取决于你是否在报告中用到）
    historical_patterns = analyze_historical_patterns(original_df)

    # 构造完整的 TimeSeriesDataFrame 并自动填补缺失
    tdf_full = TimeSeriesDataFrame.from_data_frame(
        full_df,
        id_column='item_id',
        timestamp_column='timestamp'
    ).fill_missing_values(method="auto")

    # —— 新增：准备 known_covariates ——
    # 为了在预测时满足 AutoGluon 对已知协变量索引的要求，生成未来 prediction_length 天的
    # (item_id, timestamp) 组合，并使用每个 item 最新的协变量值填充未来行。如果没有
    # 已知协变量，则传入 None。
    present_known = [c for c in KNOWN_COVARIATES if c in full_df.columns]
    if present_known:
        # 使用训练后的 predictor 生成未来索引
        future_index_df = predictor.make_future_data_frame(tdf_full)
        # 每个 item 的最新协变量值
        last_known_values = (
            full_df.sort_values('timestamp')
            .groupby('item_id')
            .last()
            .reset_index()[['item_id'] + present_known]
        )
        # 合并未来索引和最新值
        future_cov_df = future_index_df.merge(last_known_values, on='item_id', how='left')
        # 构建只包含未来期的 TimeSeriesDataFrame
        known_cov_tsd = TimeSeriesDataFrame.from_data_frame(
            future_cov_df,
            id_column='item_id',
            timestamp_column='timestamp'
        ).fill_missing_values(method="auto")
    else:
        known_cov_tsd = None
    # —— 结束 ——

    # 计算下一期时间戳
    next_date = full_df['timestamp'].max() + pd.Timedelta(days=1)
    # 用于保存 QRA 组合后的预测结果，供后续最终数字选择
    qra_at_date: Optional[pd.DataFrame] = None
    os.makedirs(out_dir, exist_ok=True)

    # 输出文件路径
    fp = os.path.join(
        out_dir,
        f'prob_forecast_optimized_PRESETS_{PRESETS}_'
        f'PREDICTION_LENGTH_{PREDICTION_LENGTH}_{next_date:%Y%m%d}.md'
    )

    # 写入表头
    header = ['model', 'timestamp', 'quantile'] + [f'pos_{i}' for i in range(1, 21)]
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')

        # ------ 高阶 QRA 组合预测 ------
        # 在写入预测结果前，先计算各模型权重和组合预测，用于后续输出
        try:
            # 计算模型权重
            weights = compute_model_weights(predictor, tdf_full)
            # 收集所有模型在 next_date 的预测
            model_preds: Dict[str, pd.DataFrame] = {}
            for mdl in predictor.model_names():
                try:
                    fc_m = predictor.predict(
                        tdf_full,
                        model=mdl,
                        known_covariates=known_cov_tsd
                    )
                    model_preds[mdl] = fc_m.xs(next_date, level='timestamp')
                except Exception as e:
                    logger.warning(f"模型 {mdl} 在 QRA 预测中失败: {e}")
            # 若成功收集到预测，则按权重组合
            if model_preds:
                # 取任意模型的列作为参考
                reference_df = next(iter(model_preds.values()))
                combined_df = pd.DataFrame(index=reference_df.index)
                for col in reference_df.columns:
                    series_sum = None
                    for mdl, df_m in model_preds.items():
                        if col not in df_m.columns:
                            continue
                        weight = weights.get(mdl, 0)
                        if series_sum is None:
                            series_sum = weight * df_m[col]
                        else:
                            series_sum += weight * df_m[col]
                    combined_df[col] = series_sum
                # 保存组合结果供后续使用
                qra_at_date = combined_df
                # 写入 QRA 组合预测结果
                qra_means = sorted(combined_df['mean'].astype(int).tolist())
                f.write(','.join(['qra_ensemble', next_date.strftime('%Y-%m-%d'), 'mean', *map(str, qra_means)]) + '\n')
                for q in QUANTILE_LEVELS:
                    vals = sorted(combined_df[str(q)].astype(int).tolist())
                    f.write(','.join(['qra_ensemble', next_date.strftime('%Y-%m-%d'), str(q), *map(str, vals)]) + '\n')
        except Exception as e:
            logger.warning(f"QRA 组合预测失败: {e}")

        # Ensemble 集成预测（简单平均）
        best_forecast = predictor.predict(
            tdf_full,
            known_covariates=known_cov_tsd
        )
        best_at_date = best_forecast.xs(next_date, level='timestamp')
        # 写入 ensemble 平均值
        means = sorted(best_at_date['mean'].astype(int).tolist())
        f.write(','.join(['ensemble', next_date.strftime('%Y-%m-%d'), 'mean', *map(str, means)]) + '\n')
        # 写入 ensemble 分位数
        for q in QUANTILE_LEVELS:
            vals = sorted(best_at_date[str(q)].astype(int).tolist())
            f.write(','.join(['ensemble', next_date.strftime('%Y-%m-%d'), str(q), *map(str, vals)]) + '\n')

        # 写入每个子模型预测
        for mdl in predictor.model_names():
            try:
                fc = predictor.predict(
                    tdf_full,
                    model=mdl,
                    known_covariates=known_cov_tsd
                )
                fc_n = fc.xs(next_date, level='timestamp')
                # 写入子模型均值
                means = sorted(fc_n['mean'].astype(int).tolist())
                f.write(','.join([mdl, next_date.strftime('%Y-%m-%d'), 'mean', *map(str, means)]) + '\n')
                # 写入子模型分位数
                for q in QUANTILE_LEVELS:
                    vals = sorted(fc_n[str(q)].astype(int).tolist())
                    f.write(','.join([mdl, next_date.strftime('%Y-%m-%d'), str(q), *map(str, vals)]) + '\n')
            except Exception as e:
                logger.warning(f"模型 {mdl} 预测失败: {e}")

    logger.info(f"下期预测文件已导出至 {fp}")

    # —— 计算并保存最终预测数字 ——
    # 如果 QRA 组合预测成功，则使用其结果进行高级数字选择，否则退回到 ensemble 平均
    # qra_at_date 是组合后的 DataFrame，其索引为每个 item_id（pos），列包括 'mean' 和各分位数
    final_pred_df: pd.DataFrame
    if qra_at_date is not None:
        final_pred_df = qra_at_date

        logger.info("使用 QRA 组合预测生成最终号码")
    else:
        final_pred_df = best_at_date
        logger.info("使用 ensemble 平均预测生成最终号码")
    # 对预测结果执行平滑与截断
    final_pred_df = postprocess_predictions(final_pred_df)
    # 使用高级数字选择算法生成最终号码
    final_numbers = compute_advanced_final_numbers(final_pred_df, historical_patterns)

    # 保存最终号码到 CSV
    final_fp = os.path.join(out_dir, f'final_prediction_{next_date:%Y%m%d}.csv')
    df_final = pd.DataFrame(
        {f'number_{i + 1}': [num] for i, num in enumerate(final_numbers)},
        index=[next_date.strftime('%Y-%m-%d')]
    )
    df_final.index.name = 'timestamp'
    df_final.to_csv(final_fp)
    logger.info(f"已保存最终预测 CSV → {final_fp}")

    # —— 导出分析报告 ——
    analysis_fp = os.path.join(out_dir, f'prediction_analysis_{next_date:%Y%m%d}.txt')
    with open(analysis_fp, 'w', encoding='utf-8') as f:
        f.write(f"KL-8 预测分析报告 - {next_date.strftime('%Y-%m-%d')}\n")
        f.write("=" * 50 + "\n\n")

        # 1. 最终预测数字
        f.write("1. 最终预测数字:\n")
        f.write(f"   {final_numbers}\n\n")

        # 2. 数字分布分析
        low_count = sum(1 for n in final_numbers if 1 <= n <= 20)
        mid_count = sum(1 for n in final_numbers if 21 <= n <= 60)
        high_count = sum(1 for n in final_numbers if 61 <= n <= 80)
        odd_count = sum(1 for n in final_numbers if n % 2 == 1)
        f.write("2. 数字分布分析:\n")
        f.write(f"   低段(1-20): {low_count} 个\n")
        f.write(f"   中段(21-60): {mid_count} 个\n")
        f.write(f"   高段(61-80): {high_count} 个\n")
        f.write(f"   奇数: {odd_count} 个, 偶数: {20 - odd_count} 个\n\n")

        # 3. 历史频率分析
        f.write("3. 历史频率分析:\n")
        hot_nums = historical_patterns['hot_numbers'][:10]
        cold_nums = historical_patterns['cold_numbers'][:10]
        f.write(f"   热号(前10): {hot_nums}\n")
        f.write(f"   冷号(前10): {cold_nums}\n\n")

        # 4. 预测置信度指标
        f.write("4. 预测置信度指标:\n")
        # 使用最终预测矩阵计算每个位置的分位数标准差以衡量不确定性
        pred_matrix = np.array([final_pred_df[str(q)].values for q in QUANTILE_LEVELS])
        pred_std = np.std(pred_matrix, axis=0)
        avg_uncertainty = np.mean(pred_std)
        f.write(f"   平均不确定性: {avg_uncertainty:.2f}\n")

    logger.info(f"已保存分析报告 → {analysis_fp}")


# ===== 回测分析 ===============================================================

def run_backtest(
        predictor: TimeSeriesPredictor,
        data_df: pd.DataFrame,
        out_dir: str,
        num_windows: int = 3
) -> Dict[str, float]:
    """执行多窗口回测以评估模型在不同时间截断下的表现。

    AutoGluon 支持通过调整 ``cutoff`` 参数对模型进行多窗口评估【655403968281618†L1070-L1101】。
    在回测过程中，我们将 ``cutoff`` 设置为负的 ``prediction_length`` 倍数，
    分别对多个切分点后的预测结果计算验证损失，并将每个窗口的得分记录下来。

    参数
    ------
    predictor: TimeSeriesPredictor
        已训练好的时间序列预测器。
    data_df: pd.DataFrame
        长格式的包含所有时间序列的 DataFrame，需要包含 ``item_id`` 和 ``timestamp``。
    out_dir: str
        保存回测结果的目录。
    num_windows: int, 默认 3
        回测窗口的数量。窗口数量越多，对模型泛化能力评估越准确，但训练数据会更少【655403968281618†L1070-L1101】。

    返回
    ------
    Dict[str, float]
        以字符串形式的 cutoff 为键，对应评估指标（例如 WQL）的值。
    """
    # 将长格式数据转换为 AutoGluon 的 TimeSeriesDataFrame
    tdf = TimeSeriesDataFrame.from_data_frame(
        data_df,
        id_column='item_id',
        timestamp_column='timestamp'
    ).fill_missing_values(method="auto")

    # 准备回测窗口
    window_scores: Dict[str, float] = {}
    # 计算 cutoff 范围：从 -(num_windows * prediction_length) 到 -prediction_length，步长为 prediction_length
    for cutoff in range(-num_windows * PREDICTION_LENGTH, 0, PREDICTION_LENGTH):
        try:
            # evaluate 方法在 cutoff 位置之后的 prediction_length 时间步上评估预测误差【655403968281618†L1070-L1101】
            score_dict = predictor.evaluate(tdf, cutoff=cutoff)
            # predictor.evaluate 返回一个字典，主评估指标存储在键与 eval_metric 相同的字段
            # 此处我们假定 eval_metric 为 'WQL' （训练时指定），因此取该字段作为窗口得分
            # 如果返回的字典包含多个指标，可根据需要调整此处逻辑
            # AutoGluon 在 v1.4.0 中通常返回一个字典，其中键为评估指标名称
            if isinstance(score_dict, dict):
                # 根据 predictor 的 eval_metric 属性尝试读取对应指标的值
                metric_name = getattr(predictor, 'eval_metric', None)
                if metric_name:
                    val = score_dict.get(str(metric_name), None)
                else:
                    # 默认情况下使用 WQL 作为评估指标
                    val = score_dict.get('WQL', None)
                if val is None and len(score_dict) > 0:
                    # 如果未找到匹配的指标，则取第一个指标值
                    val = next(iter(score_dict.values()))
            else:
                # evaluate 可能直接返回数值，直接使用
                val = score_dict

            window_scores[str(cutoff)] = float(val) if val is not None else float('nan')
            logger.info(f"回测窗口 cutoff={cutoff}: {val:.4f}")
        except Exception as e:
            logger.warning(f"回测评估 cutoff={cutoff} 失败: {e}")
            window_scores[str(cutoff)] = float('nan')

    # 保存回测结果到文件
    os.makedirs(out_dir, exist_ok=True)
    bt_fp = os.path.join(out_dir, 'backtest_results.csv')
    with open(bt_fp, 'w', encoding='utf-8') as f:
        f.write('cutoff,score\n')
        for c, s in window_scores.items():
            f.write(f"{c},{s}\n")

    logger.info("已保存回测结果 → %s", bt_fp)
    return window_scores


# ===== 主函数 ==================================================================

def main():
    """主执行函数"""
    # 保持原始路径
    data_path = '/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/kl8_order_data.csv'
    results_root = '/home/luolu/PycharmProjects/NeuralForecast/Results/kl8'

    logger.info("开始KL-8深度优化预测流程")

    # 数据加载和预处理
    logger.info("正在加载数据...")
    raw = load_data(data_path)

    logger.info("正在进行数据预处理...")
    df = preprocess_data(raw)

    logger.info("开始训练模型...")
    predictor = train_model(df)

    # 确定输出目录（基于下一期日期）
    next_period_dir = (df['timestamp'].max() + pd.Timedelta(days=1)).strftime('%Y%m%d')
    out_dir = os.path.join(results_root, next_period_dir)

    # 在生成预测之前执行回测评估
    logger.info("正在执行回测评估...")
    run_backtest(predictor, df, out_dir, num_windows=NUM_VAL_WINDOWS)

    # 生成下一期预测并导出结果
    logger.info("正在生成预测结果...")
    export_next_period_forecasts(predictor, df, raw, out_dir)

    logger.info("KL-8深度优化预测流程完成！")


if __name__ == "__main__":
    main()