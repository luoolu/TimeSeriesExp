#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KL-8 probabilistic forecasting pipeline optimized for AutoGluon 1.4."""

import os
import logging
from typing import List, Dict, Tuple, Optional
import warnings
import random

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.tabular import TabularPredictor
from autogluon.common import space

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

QUANTILE_LEVELS: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PREDICTION_LENGTH: int = 1
MAX_LAG: int = 30
PRESETS: str = "best_quality"
MODEL_PATH: str = "autogluon_kl8_model_optimized"
NUM_VAL_WINDOWS: int = 5
TIME_LIMIT: Optional[int] = 3600 * 10
KNOWN_COVARIATES: List[str] = [
    'sales_amount', 'sales_log', 'sales_pct_change',
    'pool_amount', 'pool_log', 'pool_pct_change', 'pool_to_sales'
]

HIST_WINDOW: int = 200
WEIGHT_MEAN: float = 3.0
WEIGHT_QUANTILE: float = 1.0
WEIGHT_RECENT_FREQ: float = 0.7
WEIGHT_HIST_FREQ: float = 0.3
HOT_QUANTILE: float = 0.9
COLD_QUANTILE: float = 0.1
USE_GENETIC_ALGORITHM: bool = True

USE_META_PREDICTOR: bool = True
META_MODEL_PATH: str = "autogluon_kl8_meta_model"
META_TARGET_COLS: List[str] = [
    'sum_number', 'odd_count', 'low_range_count', 'mid_range_count',
    'high_range_count', 'consecutive_count', 'mean_number'
]

HPO_NUM_TRIALS: int = 300
ENABLE_CUSTOM_HPO: bool = False

CUSTOM_HPO_SEARCH_SPACE: Dict = {
    "DeepAR": {
        "hidden_size": space.Int(20, 80),
        "num_layers": space.Int(1, 3),
        "dropout_rate": space.Categorical(0.1, 0.2, 0.3),
    },
    "TemporalFusionTransformer": {
        "hidden_size": space.Int(16, 64),
        "dropout_rate": space.Categorical(0.1, 0.2, 0.3),
    },
    "PatchTST": {
        "patch_size": space.Categorical(4, 8, 16),
        "hidden_size": space.Int(16, 64),
    },
    "ChronosModel": {
        "model_path": space.Categorical(
            "autogluon/chronos-bolt-tiny",
            "autogluon/chronos-bolt-mini",
            "autogluon/chronos-bolt-small",
            "autogluon/chronos-bolt-base",
        ),
    },
    "PerStepTabularModel": {
        "model_name": space.Categorical("GBM", "CAT", "RF"),
        "max_num_items": 20000,
        "max_num_samples": 1000000,
    },
    "RecursiveTabularModel": {
        "model_name": space.Categorical("GBM", "CAT"),
        "lags": space.Categorical(
            [1, 2, 3],
            [1, 2, 3, 7],
            [1, 2, 3, 7, 14]
        ),
        "target_scaler": space.Categorical("standard", "mean_abs"),
    },
    "TiDEModel": {
        "context_length": space.Int(64, 256),
        "num_layers_encoder": space.Int(1, 3),
        "num_layers_decoder": space.Int(1, 3),
        "dropout_rate": space.Categorical(0.1, 0.2, 0.3),
    },
    "WaveNetModel": {
        "num_residual_channels": space.Int(16, 64),
        "num_skip_channels": space.Int(16, 64),
        "dilation_depth": space.Int(1, 6),
        "num_stacks": space.Int(1, 2),
    },
}

HYPERPARAMETERS_TUNING: Optional[Dict] = CUSTOM_HPO_SEARCH_SPACE if ENABLE_CUSTOM_HPO else None

HYPERPARAMETER_TUNE_KWARGS: Optional[Dict[str, any]] = (
    {
        "num_trials": HPO_NUM_TRIALS,
        "scheduler": "local",
        "searcher": "random",
    }
    if ENABLE_CUSTOM_HPO
    else None
)

WEIGHT_UNCERTAINTY: float = 0.5
QUANTILE_SCORE_WEIGHTS: Dict[str, float] = {
    '0.1': 1.0,
    '0.2': 1.0,
    '0.3': 1.0,
    '0.4': 1.0,
    '0.5': 1.0,
    '0.6': 1.0,
    '0.7': 1.0,
    '0.8': 1.0,
    '0.9': 1.0,
}

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = "/home/luolu/PycharmProjects/TimeSeriesExp/GetData/kl8/kl8_order_data.csv"
DATA_PATH_STR = os.environ.get("KL8_DATA_PATH", str(DEFAULT_DATA_PATH))
DATA_PATH = Path(DATA_PATH_STR)

if not DATA_PATH.exists():
    logger.warning(f"路径 {DATA_PATH} 不存在。正在尝试默认路径 {DEFAULT_DATA_PATH}...")
    if DEFAULT_DATA_PATH.exists():
        DATA_PATH = DEFAULT_DATA_PATH
        logger.info(f"已成功回退到默认数据路径: {DATA_PATH}")
    else:
        logger.error(f"默认数据路径 {DEFAULT_DATA_PATH} 也未找到。请检查您的文件结构。")

DEFAULT_RESULTS_ROOT = "/home/luolu/PycharmProjects/TimeSeriesExp/NeuralForecast/Results/kl8"
RESULTS_ROOT_STR = os.environ.get("KL8_RESULTS_ROOT", str(DEFAULT_RESULTS_ROOT))
RESULTS_ROOT = Path(RESULTS_ROOT_STR)

TREND_WINDOW: int = HIST_WINDOW
WEIGHT_TREND: float = 0.7
CANDIDATE_POOL_SIZE: int = 400
HYPERPARAMETERS: Optional[Dict] = None

def compute_model_weights(predictor: TimeSeriesPredictor, tdf: TimeSeriesDataFrame) -> Dict[str, float]:
    """Compute normalized leaderboard-based weights for ensemble blending."""
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

def load_data(path: Path) -> pd.DataFrame:
    """加载KL-8原始数据并进行基础清理"""
    if not path.exists():
        logger.error(f"数据文件未找到: {path}")
        raise FileNotFoundError(f"Data file not found at {path}")

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

    df['day_of_week'] = df['timestamp'].dt.dayofweek.astype('int8')
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype('int16')
    df['month'] = df['timestamp'].dt.month.astype('int8')
    df['day_of_month'] = df['timestamp'].dt.day.astype('int8')
    df['quarter'] = df['timestamp'].dt.quarter.astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')

    df['day_of_year'] = df['timestamp'].dt.dayofyear.astype('int16')
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    num_cols = [f'开奖号_{i}' for i in range(1, 21)]
    if all(col in df.columns for col in num_cols):
        numbers_array = df[num_cols].values

        df['mean_number'] = np.mean(numbers_array, axis=1)
        df['std_number'] = np.std(numbers_array, axis=1)
        df['median_number'] = np.median(numbers_array, axis=1)
        df['range_number'] = np.ptp(numbers_array, axis=1)
        df['sum_number'] = np.sum(numbers_array, axis=1)

        df['skewness'] = stats.skew(numbers_array, axis=1)
        df['kurtosis'] = stats.kurtosis(numbers_array, axis=1)

        df['low_range_count'] = np.sum((numbers_array >= 1) & (numbers_array <= 20), axis=1)
        df['mid_range_count'] = np.sum((numbers_array >= 21) & (numbers_array <= 60), axis=1)
        df['high_range_count'] = np.sum((numbers_array >= 61) & (numbers_array <= 80), axis=1)

        df['odd_count'] = np.sum(numbers_array % 2 == 1, axis=1)
        df['even_count'] = np.sum(numbers_array % 2 == 0, axis=1)

        sorted_numbers = np.sort(numbers_array, axis=1)
        consecutive_diffs = np.diff(sorted_numbers, axis=1)
        df['consecutive_count'] = np.sum(consecutive_diffs == 1, axis=1)
        df['max_gap'] = np.max(consecutive_diffs, axis=1)
        df['min_gap'] = np.min(consecutive_diffs, axis=1)

        try:
            # 计算相邻间隔的均值与标准差
            df['gap_mean'] = np.mean(consecutive_diffs, axis=1)
            df['gap_std'] = np.std(consecutive_diffs, axis=1)
            # 偏度和峰度可能因数据规模小导致警告，使用 nan_policy='propagate'
            df['gap_skewness'] = stats.skew(consecutive_diffs, axis=1, nan_policy='propagate')
            df['gap_kurtosis'] = stats.kurtosis(consecutive_diffs, axis=1, nan_policy='propagate')
        except Exception:
            # 若计算失败，则填充缺失值
            df['gap_mean'] = np.nan
            df['gap_std'] = np.nan
            df['gap_skewness'] = np.nan
            df['gap_kurtosis'] = np.nan

        primes = np.array([
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
            43, 47, 53, 59, 61, 67, 71, 73, 79
        ], dtype=int)
        is_prime = np.isin(numbers_array, primes)
        df['prime_count'] = is_prime.sum(axis=1)
        prime_values = numbers_array * is_prime
        df['prime_sum'] = prime_values.sum(axis=1)

        df['sum_top5'] = np.sum(sorted_numbers[:, -5:], axis=1)
        df['sum_bottom5'] = np.sum(sorted_numbers[:, :5], axis=1)

        try:
            scaler = StandardScaler()
            numbers_scaled = scaler.fit_transform(numbers_array)
            km = KMeans(n_clusters=5, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(numbers_scaled)
            df['num_cluster'] = cluster_labels.astype('int8')
        except Exception:
            # 如果聚类失败，填充0作为默认类别
            df['num_cluster'] = 0

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

        # 1. 完全平方数和完全立方数计数
        squares = np.array([n for n in range(1, 81) if int(np.sqrt(n)) ** 2 == n], dtype=int)
        cubes = np.array([1, 8, 27, 64], dtype=int)
        is_square = np.isin(numbers_array, squares)
        is_cube = np.isin(numbers_array, cubes)
        df['square_count'] = is_square.sum(axis=1)
        df['cube_count'] = is_cube.sum(axis=1)
        df['multiple_of_3_count'] = np.sum(numbers_array % 3 == 0, axis=1)
        # 公式：(n - 1) % 9 + 1 对 n>0 恒成立；
        digital_roots = ((numbers_array - 1) % 9) + 1
        df['dr_mean'] = np.mean(digital_roots, axis=1)
        df['dr_std'] = np.std(digital_roots, axis=1)
        for d in range(1, 10):
            df[f'dr_count_{d}'] = (digital_roots == d).sum(axis=1)
        total_sum = np.sum(numbers_array, axis=1)
        df['digital_root_sum'] = ((total_sum - 1) % 9) + 1

        try:
            centered_numbers = numbers_array - numbers_array.mean(axis=1, keepdims=True)
            fft_vals = np.abs(np.fft.rfft(centered_numbers, axis=1))
            fft_power = np.square(fft_vals)
            power_sum = fft_power.sum(axis=1, keepdims=True)
            power_sum[power_sum == 0] = 1.0
            normalized_power = fft_power / power_sum

            samples = normalized_power.shape[0]
            df['fft_energy_ratio_1'] = np.zeros(samples)
            df['fft_energy_ratio_2'] = np.zeros(samples)
            df['fft_peak_index'] = np.zeros(samples, dtype='int8')

            if normalized_power.shape[1] > 1:
                main_band = normalized_power[:, 1:4]  # 聚焦前3个频率分量
                df.loc[:, 'fft_energy_ratio_1'] = main_band[:, 0]
                if main_band.shape[1] > 1:
                    df.loc[:, 'fft_energy_ratio_2'] = main_band[:, 1]
                peak_indices = np.argmax(normalized_power[:, 1:], axis=1) + 1
                df.loc[:, 'fft_peak_index'] = peak_indices.astype('int8')

            # 频谱熵：衡量能量分布的不确定度
            spectral_entropy = stats.entropy(normalized_power, axis=1)
            df['fft_spectral_entropy'] = spectral_entropy
        except Exception:
            df['fft_energy_ratio_1'] = 0.0
            df['fft_energy_ratio_2'] = 0.0
            df['fft_peak_index'] = 0
            df['fft_spectral_entropy'] = 0.0

    # 部分数据集中包含《本期销售金额》和《选十玩法奖池金额》列，通常带有千位分隔符（逗号），需要转为数值。
    if '本期销售金额' in df.columns:
        # 将逗号（中英文）移除后转换为浮点型
        sales_raw = df['本期销售金额'].astype(str).str.replace(',', '', regex=False).str.replace('，', '', regex=False)
        df['sales_amount'] = pd.to_numeric(sales_raw, errors='coerce')
    if 'sales_amount' in df.columns:
        # 对销售额取对数，加入环比变化率
        df['sales_log'] = np.log1p(df['sales_amount'])
        df['sales_pct_change'] = df['sales_amount'].pct_change().fillna(0)

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
    df = extract_additional_features(df)
    return df

def extract_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive order, digit and overlap descriptors for each draw."""
    df = df.copy()

    # 确定号码列和顺序列
    num_cols = [f'开奖号_{i}' for i in range(1, 21) if f'开奖号_{i}' in df.columns]
    order_cols = [f'出球顺序_{i}' for i in range(1, 21) if f'出球顺序_{i}' in df.columns]

    if not num_cols:
        return df

    numbers_array = df[num_cols].to_numpy(dtype=float)

    if order_cols and len(num_cols) == len(order_cols):
        # 逐行计算顺序差异
        def compute_order_stats(row):
            try:
                nums = [int(v) for v in row[num_cols]]
                orders = [int(v) for v in row[order_cols]]
                asc_map = {n: i for i, n in enumerate(sorted(nums))}
                order_map = {n: i for i, n in enumerate(orders)}
                diffs = [order_map.get(n, np.nan) - asc_map.get(n, np.nan) for n in nums]
                diffs_arr = np.array(diffs, dtype=float)
                return pd.Series({
                    'order_diff_mean': float(np.nanmean(diffs_arr)),
                    'order_diff_std': float(np.nanstd(diffs_arr)),
                    'order_diff_max': float(np.nanmax(diffs_arr)),
                    'order_diff_min': float(np.nanmin(diffs_arr)),
                    'order_diff_pos_count': float(np.nansum(diffs_arr > 0)),
                    'order_diff_neg_count': float(np.nansum(diffs_arr < 0)),
                })
            except Exception:
                return pd.Series({k: np.nan for k in [
                    'order_diff_mean', 'order_diff_std', 'order_diff_max',
                    'order_diff_min', 'order_diff_pos_count', 'order_diff_neg_count'
                ]})
        order_stats_df = df.apply(compute_order_stats, axis=1)
        df = pd.concat([df, order_stats_df], axis=1)

    tens = (numbers_array // 10).astype(float)
    units = (numbers_array % 10).astype(float)
    digitsum = tens + units
    df['tens_mean'] = np.mean(tens, axis=1)
    df['tens_std'] = np.std(tens, axis=1)
    df['units_mean'] = np.mean(units, axis=1)
    df['units_std'] = np.std(units, axis=1)
    df['digitsum_mean'] = np.mean(digitsum, axis=1)
    df['digitsum_std'] = np.std(digitsum, axis=1)
    df['repdigit_count'] = (tens == units).sum(axis=1).astype(int)

    deltas = np.diff(numbers_array, axis=0, prepend=np.full((1, numbers_array.shape[1]), np.nan))
    df['pos_deltas_mean'] = np.nanmean(deltas, axis=1)
    df['pos_deltas_std'] = np.nanstd(deltas, axis=1)
    df['pos_deltas_positive'] = (deltas > 0).sum(axis=1).astype(float)
    df['pos_deltas_negative'] = (deltas < 0).sum(axis=1).astype(float)

    df['group1_sum'] = np.sum(numbers_array[:, 0:5], axis=1)
    df['group1_mean'] = np.mean(numbers_array[:, 0:5], axis=1)
    df['group1_std'] = np.std(numbers_array[:, 0:5], axis=1)
    df['group2_sum'] = np.sum(numbers_array[:, 5:10], axis=1)
    df['group2_mean'] = np.mean(numbers_array[:, 5:10], axis=1)
    df['group2_std'] = np.std(numbers_array[:, 5:10], axis=1)
    df['group3_sum'] = np.sum(numbers_array[:, 10:15], axis=1)
    df['group3_mean'] = np.mean(numbers_array[:, 10:15], axis=1)
    df['group3_std'] = np.std(numbers_array[:, 10:15], axis=1)
    df['group4_sum'] = np.sum(numbers_array[:, 15:20], axis=1)
    df['group4_mean'] = np.mean(numbers_array[:, 15:20], axis=1)
    df['group4_std'] = np.std(numbers_array[:, 15:20], axis=1)
    df['first_half_sum'] = np.sum(numbers_array[:, 0:10], axis=1)
    df['second_half_sum'] = np.sum(numbers_array[:, 10:20], axis=1)
    df['half_sum_diff'] = df['first_half_sum'] - df['second_half_sum']
    num_sets = [set(row) for row in numbers_array]
    df['overlap_prev1'] = [len(num_sets[i] & num_sets[i-1]) if i > 0 else 0 for i in range(len(num_sets))]
    df['overlap_prev2'] = [len(num_sets[i] & num_sets[i-2]) if i > 1 else 0 for i in range(len(num_sets))]
    df['overlap_prev1_ratio'] = df['overlap_prev1'] / 20.0
    df['overlap_prev2_ratio'] = df['overlap_prev2'] / 20.0

    return df

def calculate_omission_features(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """
    新增优化：计算每个数字的遗漏期数特征。
    为1-80中的每个数字计算自上次出现以来经过的期数。
    """
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    omission_cols = {f'omission_{i}': [] for i in range(1, 81)}
    last_seen = {i: -1 for i in range(1, 81)}

    for index, row in df.iterrows():
        current_numbers = set(row[col] for col in num_cols if pd.notna(row[col]))

        for i in range(1, 81):
            if i in current_numbers:
                last_seen[i] = index

            # 遗漏期数 = 当前期索引 - 上次出现期索引
            omission_value = index - last_seen[i] if last_seen[i] != -1 else index + 1
            omission_cols[f'omission_{i}'].append(omission_value)

    omission_df = pd.DataFrame(omission_cols)
    return pd.concat([df, omission_df], axis=1)

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

def add_ewm_features(df: pd.DataFrame, cols: List[str], spans: List[int]) -> pd.DataFrame:
    """为指定列添加指数加权移动平均特征，捕捉不同时间尺度的趋势。"""
    if not cols or not spans:
        return df

    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        if not np.issubdtype(series.dtype, np.number):
            continue
        for span in spans:
            if span <= 1:
                continue
            new_col = f'{col}_ewm_span_{span}'
            df[new_col] = series.ewm(span=span, adjust=False).mean()
    return df

def create_lag_features(df: pd.DataFrame, num_cols: List[str], lags: List[int]) -> pd.DataFrame:
    """创建滞后特征"""
    df = df.copy()

    for col in num_cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

            for window in [3, 7, 14]:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window).max()
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window).min()

    return df

def compute_item_static_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """根据每个 item 的历史行为构建静态特征，供 AutoGluon 使用。"""
    if 'item_id' not in long_df.columns or 'target' not in long_df.columns:
        return pd.DataFrame()

    grouped = long_df.groupby('item_id')
    total_positions = max(grouped.ngroups, 1)
    denom = max(total_positions - 1, 1)

    records = []
    for item_id, group in grouped:
        try:
            position_idx = int(str(item_id).split('_')[-1])
        except (ValueError, IndexError):
            position_idx = 0

        series = group.sort_values('timestamp')['target'].astype(float)
        values = series.to_numpy()

        recent_window = min(20, len(series))
        early_window = min(20, len(series))

        recent_mean = series.tail(recent_window).mean() if recent_window else np.nan
        early_mean = series.head(early_window).mean() if early_window else np.nan
        trend = (recent_mean - early_mean) if recent_window and early_window else series.diff().mean()

        low_ratio = float(np.mean(values <= 20)) if len(values) else 0.0
        mid_ratio = float(np.mean((values > 20) & (values <= 60))) if len(values) else 0.0
        high_ratio = float(np.mean(values > 60)) if len(values) else 0.0

        autocorr_lag1 = series.autocorr(lag=1) if len(series) > 1 else 0.0

        record = {
            'item_id': item_id,
            'position_index': position_idx,
            'position_norm': (position_idx - 1) / denom if denom else 0.0,
            'position_sin': np.sin(2 * np.pi * (position_idx - 1) / total_positions),
            'position_cos': np.cos(2 * np.pi * (position_idx - 1) / total_positions),
            'is_first_half': int(position_idx <= total_positions / 2),
            'is_even_position': int(position_idx % 2 == 0),
            'hist_target_mean': series.mean(),
            'hist_target_std': series.std(),
            'hist_target_median': series.median(),
            'hist_target_last': series.iloc[-1] if len(series) else np.nan,
            'hist_target_trend': trend if not np.isnan(trend) else 0.0,
            'hist_target_q25': series.quantile(0.25),
            'hist_target_q75': series.quantile(0.75),
            'hist_low_ratio': low_ratio,
            'hist_mid_ratio': mid_ratio,
            'hist_high_ratio': high_ratio,
            'hist_autocorr_lag1': autocorr_lag1 if not np.isnan(autocorr_lag1) else 0.0,
        }

        records.append(record)

    static_df = pd.DataFrame(records).set_index('item_id')
    static_df.index = static_df.index.astype(str)
    static_df = static_df.fillna(0.0)
    return static_df


def prepare_static_features_for_ag(
    static_features: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """Ensure static feature frame exposes an explicit string `item_id` column."""
    if static_features is None or static_features.empty:
        return None

    static_df = static_features.copy()
    if 'item_id' not in static_df.columns:
        index_name = getattr(static_df.index, 'name', None)
        static_df = static_df.reset_index()
        if 'item_id' not in static_df.columns:
            rename_source = index_name if index_name else static_df.columns[0]
            static_df = static_df.rename(columns={rename_source: 'item_id'})

    static_df['item_id'] = static_df['item_id'].astype(str)
    return static_df

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """深度优化的数据预处理，返回长格式数据和静态特征矩阵"""
    df = extract_advanced_features(df)

    num_cols = [f'开奖号_{i}' for i in range(1, 21)]

    logger.info("正在计算遗漏特征...")
    df = calculate_omission_features(df, num_cols)

    # 异常值处理
    df = detect_and_handle_anomalies(df, num_cols)

    df = smooth_numeric_series(df, num_cols)

    lags = [1, 2, 3, 7, 14, 21]
    extra_lag_cols: List[str] = []
    for col in [
        'sales_amount', 'sales_log', 'sales_pct_change',
        'pool_amount', 'pool_log', 'pool_pct_change', 'pool_to_sales',
        'mean_diff', 'sum_diff', 'range_diff', 'odd_diff', 'even_diff'
    ]:
        if col in df.columns:
            extra_lag_cols.append(col)

    for col in [
        'group1_sum', 'group1_mean', 'group1_std',
        'group2_sum', 'group2_mean', 'group2_std',
        'group3_sum', 'group3_mean', 'group3_std',
        'group4_sum', 'group4_mean', 'group4_std',
        'first_half_sum', 'second_half_sum', 'half_sum_diff',
        'overlap_prev1', 'overlap_prev2', 'overlap_prev1_ratio', 'overlap_prev2_ratio'
    ]:
        if col in df.columns:
            extra_lag_cols.append(col)
    df = create_lag_features(df, num_cols + extra_lag_cols, lags)

    ewm_candidate_cols = [
        'sum_number', 'mean_number', 'range_number', 'std_number',
        'odd_count', 'even_count', 'low_range_count', 'mid_range_count', 'high_range_count',
        'prime_count', 'prime_sum', 'square_count', 'cube_count',
        'multiple_of_3_count', 'digital_root_sum', 'consecutive_count', 'gap_mean',
        'first_half_sum', 'second_half_sum', 'half_sum_diff',
        'sales_amount', 'sales_log', 'sales_pct_change',
        'pool_amount', 'pool_log', 'pool_pct_change', 'pool_to_sales'
    ]
    ewm_cols = [col for col in ewm_candidate_cols if col in df.columns]
    df = add_ewm_features(df, ewm_cols, spans=[3, 7, 14, 28])

    covariate_cols = [
        'day_of_week', 'week_of_year', 'month', 'day_of_month', 'quarter', 'is_weekend',
        'day_of_year', 'week_sin', 'week_cos', 'month_sin', 'month_cos',
        'sales_amount', 'sales_log', 'sales_pct_change',
        'pool_amount', 'pool_log', 'pool_pct_change', 'pool_to_sales',
        'mean_number', 'std_number', 'median_number', 'range_number', 'sum_number',
        'skewness', 'kurtosis', 'low_range_count', 'mid_range_count', 'high_range_count',
        'odd_count', 'even_count', 'consecutive_count', 'max_gap', 'min_gap',
        'prime_count', 'prime_sum', 'sum_top5', 'sum_bottom5', 'num_cluster',
        'mean_diff', 'sum_diff', 'range_diff', 'odd_diff', 'even_diff',
        'order_diff_mean', 'order_diff_std', 'order_diff_max', 'order_diff_min',
        'order_diff_pos_count', 'order_diff_neg_count',
        'tens_mean', 'tens_std', 'units_mean', 'units_std',
        'digitsum_mean', 'digitsum_std', 'repdigit_count',
        'pos_deltas_mean', 'pos_deltas_std', 'pos_deltas_positive', 'pos_deltas_negative'
    ]
    covariate_cols.extend([
        'square_count', 'cube_count', 'multiple_of_3_count',
        'dr_mean', 'dr_std', 'digital_root_sum'
    ])
    covariate_cols.extend([f'dr_count_{i}' for i in range(1, 10)])
    omission_feature_cols = [f'omission_{i}' for i in range(1, 81)]
    covariate_cols.extend(omission_feature_cols)

    covariate_cols.extend([
        'group1_sum', 'group1_mean', 'group1_std',
        'group2_sum', 'group2_mean', 'group2_std',
        'group3_sum', 'group3_mean', 'group3_std',
        'group4_sum', 'group4_mean', 'group4_std',
        'first_half_sum', 'second_half_sum', 'half_sum_diff',
        'overlap_prev1', 'overlap_prev2', 'overlap_prev1_ratio', 'overlap_prev2_ratio'
    ])

    covariate_cols = [col for col in covariate_cols if col in df.columns]

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
    out['item_id'] = out['item_id'].astype(str)

    numeric_features = out.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['target']]

    # 把 ±inf 转成 NaN，避免后续 scaler 报错
    out = out.replace([np.inf, -np.inf], np.nan)

    # 填充缺失值（用中位数）
    for col in numeric_features:
        if out[col].isnull().sum() > 0:
            out[col] = out[col].fillna(out[col].median())

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

    static_features = compute_item_static_features(out)

    logger.info(
        "预处理完成：%d 条记录，%d 个动态特征，%d 个静态特征",
        len(out),
        len(numeric_features),
        static_features.shape[1] if not static_features.empty else 0
    )

    return out.reset_index(drop=True), static_features

def train_model(
    train_df: pd.DataFrame,
    static_features: Optional[pd.DataFrame] = None
) -> TimeSeriesPredictor:
    # Convert to TimeSeriesDataFrame
    static_features_df = prepare_static_features_for_ag(static_features)
    tdf = TimeSeriesDataFrame.from_data_frame(
        train_df,
        id_column='item_id',
        timestamp_column='timestamp',
        static_features_df=static_features_df
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
        fit_kwargs['hyperparameter_tune_kwargs'] = HYPERPARAMETER_TUNE_KWARGS
        # 当进行超参搜索时，可保持 ensemble 启用以集成不同超参版本的模型
    else:
        logger.info("启用 AutoGluon best_quality 预设的完整模型组")

    static_feat_count = static_features.shape[1] if static_features is not None else 0
    logger.info("开始模型训练… (静态特征数: %d)", static_feat_count)
    predictor.fit(**fit_kwargs)
    logger.info("模型训练完成")

    return predictor

def train_and_predict_meta_features(
    original_df: pd.DataFrame
) -> Optional[Dict[str, float]]:
    """
    训练一个TabularPredictor来预测下一期的宏观统计特征，作为动态约束。
    """
    if not USE_META_PREDICTOR:
        return None

    logger.info("开始训练元特征预测器...")

    # 确保目标列存在
    targets = [col for col in META_TARGET_COLS if col in original_df.columns]
    if not targets:
        logger.warning("元特征目标列不存在，跳过元模型训练。")
        return None

    df_meta = original_df.copy()

    predictor_cols = []
    lags = [1, 2, 3, 7, 14]
    windows = [3, 7, 14]

    for col in targets:
        for lag in lags:
            fname = f'{col}_lag_{lag}'
            df_meta[fname] = df_meta[col].shift(lag)
            predictor_cols.append(fname)
        for w in windows:
            fname_mean = f'{col}_roll_mean_{w}'
            fname_std = f'{col}_roll_std_{w}'
            df_meta[fname_mean] = df_meta[col].rolling(window=w).mean()
            df_meta[fname_std] = df_meta[col].rolling(window=w).std()
            predictor_cols.extend([fname_mean, fname_std])

    # 移除因滞后产生的NaN
    df_meta = df_meta.dropna(subset=predictor_cols).reset_index(drop=True)

    predictions = {}

    # 为每个元目标训练一个模型
    for target in targets:
        logger.info(f"正在为元目标 '{target}' 训练模型...")
        # 准备训练数据和预测数据
        train_data = df_meta.drop(columns=[target]).iloc[:-1]
        predict_data = df_meta.drop(columns=[target]).iloc[-1:]

        # 定义此目标的标签列
        train_data['label'] = df_meta[target].iloc[:-1]

        # 训练TabularPredictor
        meta_predictor = TabularPredictor(
            label='label',
            path=os.path.join(META_MODEL_PATH, target),
            problem_type='regression',
            eval_metric='root_mean_squared_error'
        ).fit(
            train_data,
            presets='medium_quality',
            time_limit=600  # 每个元模型最多训练10分钟
        )

        # 预测下一期
        pred = meta_predictor.predict(predict_data)[0]
        predictions[f'predicted_{target}'] = float(pred)
        logger.info(f"'{target}' 的预测值为: {pred:.2f}")

    logger.info("元特征预测完成。")
    return predictions

def analyze_historical_patterns(df: pd.DataFrame) -> Dict[str, any]:
    """分析历史数字模式"""
    df_sorted = df.sort_values('timestamp')
    num_cols = [f'开奖号_{i}' for i in range(1, 21)]
    # 全局频率
    all_numbers = df_sorted[num_cols].values.flatten()
    number_freq = pd.Series(all_numbers).value_counts().sort_index()
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
        low_counts, mid_counts, high_counts, odd_counts, sums, consecutive = [], [], [], [], [], []
        for nums in dataframe[num_cols].values:
            low_counts.append(sum(1 for n in nums if 1 <= n <= 20))
            mid_counts.append(sum(1 for n in nums if 21 <= n <= 60))
            high_counts.append(sum(1 for n in nums if 61 <= n <= 80))
            odd_counts.append(sum(1 for n in nums if n % 2 == 1))
            sums.append(sum(nums))
            sorted_nums = sorted(nums)
            consecutive.append(sum(1 for i in range(len(sorted_nums) - 1) if sorted_nums[i+1] - sorted_nums[i] == 1))

        return {
            'low': float(np.mean(low_counts)),
            'mid': float(np.mean(mid_counts)),
            'high': float(np.mean(high_counts)),
            'odd': float(np.mean(odd_counts)),
            'sum': float(np.mean(sums)),
            'consecutive': float(np.mean(consecutive))
        }
    dist_stats_global = compute_distribution_stats(df_sorted)
    dist_stats_recent = compute_distribution_stats(recent_df)
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
        original_df: Optional[pd.DataFrame] = None,
        dynamic_constraints: Optional[Dict[str, float]] = None
) -> List[int]:
    """Score predictions with historical context and return 20 selected numbers."""
    # 初始化分数
    number_scores: Dict[int, float] = {n: 0.0 for n in range(1, 81)}

    #
    trending_scores: Dict[int, float] = {n: 0.0 for n in range(1, 81)}
    if original_df is not None and TREND_WINDOW > 0:
        try:
            # 确保按时间排序
            df_sorted = original_df.sort_values('timestamp') if 'timestamp' in original_df.columns else original_df.copy()
            # 号码列
            num_cols = [f'开奖号_{i}' for i in range(1, 21) if f'开奖号_{i}' in df_sorted.columns]
            total_length = len(df_sorted)
            recent_df = df_sorted.iloc[-TREND_WINDOW:]
            prev_df = df_sorted.iloc[-2 * TREND_WINDOW:-TREND_WINDOW] if total_length >= 2 * TREND_WINDOW else df_sorted.iloc[:-TREND_WINDOW]
            recent_counts = pd.Series(recent_df[num_cols].values.flatten()).value_counts()
            prev_counts = pd.Series(prev_df[num_cols].values.flatten()).value_counts() if len(prev_df) > 0 else pd.Series(dtype=int)
            # 标准化差值到 [-1, 1] 区间
            diffs = {}
            all_nums = set(range(1, 81))
            for n in all_nums:
                r = recent_counts.get(n, 0)
                p = prev_counts.get(n, 0)
                diff = r - p
                diffs[n] = diff
            # 归一化到 [-1, 1]
            max_abs_diff = max(abs(v) for v in diffs.values()) if diffs else 1.0
            if max_abs_diff == 0:
                max_abs_diff = 1.0
            for n, d in diffs.items():
                trending_scores[n] = float(d) / max_abs_diff
        except Exception:
            trending_scores = {n: 0.0 for n in range(1, 81)}
    # 1. 预测评分：均值和分位数
    # 使用灵活的权重机制：均值列使用 WEIGHT_MEAN，分位列根据 QUANTILE_SCORE_WEIGHTS
    # 指定的权重进行加权。如果某个分位点未设置则使用 WEIGHT_QUANTILE。
    pred_columns = ['mean'] + [str(q) for q in QUANTILE_LEVELS]
    for col in pred_columns:
        if col not in predictions.columns:
            continue
        # 根据列名选择权重
        if col == 'mean':
            weight = WEIGHT_MEAN
        else:
            # 从字典中获取分位点的权重，不存在则使用默认值
            weight = QUANTILE_SCORE_WEIGHTS.get(col, WEIGHT_QUANTILE)
        for val in predictions[col]:
            num = int(np.clip(round(float(val)), 1, 80))
            number_scores[num] += weight
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

    for num in number_scores:
        number_scores[num] += WEIGHT_TREND * trending_scores.get(num, 0.0)
    # 保证得分非负
    for num in number_scores:
        if number_scores[num] < 0:
            number_scores[num] = 0.0
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
    sorted_scores = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_count = max(30, min(CANDIDATE_POOL_SIZE, len(sorted_scores)))
    candidate_numbers = [num for num, _ in sorted_scores[:candidate_count]]

    constraints = {}
    if dynamic_constraints:
        constraints['low'] = dynamic_constraints.get('predicted_low_range_count', historical_patterns['dist_stats']['low'])
        constraints['mid'] = dynamic_constraints.get('predicted_mid_range_count', historical_patterns['dist_stats']['mid'])
        constraints['high'] = dynamic_constraints.get('predicted_high_range_count', historical_patterns['dist_stats']['high'])
        constraints['odd'] = dynamic_constraints.get('predicted_odd_count', historical_patterns['dist_stats']['odd'])
        constraints['sum'] = dynamic_constraints.get('predicted_sum_number', historical_patterns['dist_stats']['sum'])
        constraints['consecutive'] = dynamic_constraints.get('predicted_consecutive_count', historical_patterns['dist_stats']['consecutive'])
    else:
        constraints = historical_patterns['dist_stats']

    # 根据配置决定使用哪种优化策略
    if USE_GENETIC_ALGORITHM:
        # 使用遗传算法优化选择
        final_numbers = genetic_algorithm_selection(candidate_numbers, number_scores, constraints)
    else:
        # 使用规则型分布约束优化
        final_numbers = optimize_number_selection(candidate_numbers, constraints)
    return sorted(final_numbers)

def optimize_number_selection(candidates: List[int], constraints: Dict[str, any]) -> List[int]:
    """根据历史分布约束从候选数字中选出20个号码"""
    # 目标分布
    target_low = max(0, min(20, int(round(constraints.get('low', 5.0)))))
    target_mid = max(0, min(20, int(round(constraints.get('mid', 10.0)))))
    target_high = max(0, min(20, int(round(constraints.get('high', 5.0)))))
    # 调整使三者和为20
    total = target_low + target_mid + target_high
    if total != 20:
        target_mid += (20 - total)
        target_mid = max(0, target_mid)
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
    target_odd = int(round(constraints.get('odd', 10.0)))
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

def genetic_algorithm_selection(candidates: List[int], number_scores: Dict[int, float],
                               constraints: Dict[str, any],
                               population_size: int = 300,
                               generations: int = 150,
                               crossover_rate: float = 0.9,
                               mutation_rate: float = 0.4) -> List[int]:
    """
    使用简单遗传算法从候选数字中选择20个号码，适应度函数考虑预测评分和多重（动态）分布约束。
    参数已调优以进行更广泛的搜索。
    """
    # 目标约束
    target_low = int(round(constraints.get('low', 5.0)))
    target_mid = int(round(constraints.get('mid', 10.0)))
    target_high = int(round(constraints.get('high', 5.0)))
    target_odd = int(round(constraints.get('odd', 10.0)))
    target_sum = int(round(constraints.get('sum', 810.0)))
    target_consecutive = int(round(constraints.get('consecutive', 1.0)))

    # 适应度函数：总评分减去分布偏差惩罚
    def fitness(individual: List[int]) -> float:
        score_sum = sum(number_scores.get(n, 0) for n in individual)

        low = sum(1 for n in individual if 1 <= n <= 20)
        mid = sum(1 for n in individual if 21 <= n <= 60)
        high = sum(1 for n in individual if 61 <= n <= 80)
        odd = sum(1 for n in individual if n % 2 == 1)
        current_sum = sum(individual)

        # 计算连号
        sorted_ind = sorted(individual)
        consecutive = sum(1 for i in range(len(sorted_ind) - 1) if sorted_ind[i+1] - sorted_ind[i] == 1)

        # 计算惩罚项
        penalty = (
            abs(low - target_low) * 0.1 +
            abs(mid - target_mid) * 0.1 +
            abs(high - target_high) * 0.1 +
            abs(odd - target_odd) * 0.05 +
            abs(current_sum - target_sum) * 0.002 +  # 对总和偏差进行惩罚
            abs(consecutive - target_consecutive) * 0.2 # 对连号偏差进行惩罚
        )
        return score_sum - penalty

    # 初始种群
    def random_individual():
        return sorted(random.sample(candidates, 20))
    population = [random_individual() for _ in range(population_size)]

    # 迭代
    for _ in range(generations):
        fitness_values = [fitness(ind) for ind in population]

        # 选择：精英保留 + 轮盘赌选择
        elite_num = max(2, int(population_size * 0.1))
        elite_indices = sorted(range(population_size), key=lambda i: fitness_values[i], reverse=True)[:elite_num]
        elites = [population[i] for i in elite_indices]

        new_population = elites[:]

        # 生成后代
        while len(new_population) < population_size:
            # 交叉
            if random.random() < crossover_rate:
                p1, p2 = random.choices(population, weights=fitness_values, k=2)
                point = random.randint(1, 19)
                child_set = set(p1[:point]) | set(p2[point:])
                # 补全或截断到20个
                while len(child_set) < 20 and len(child_set) < len(candidates):
                    child_set.add(random.choice(candidates))
                child = sorted(list(child_set))[:20]
                new_population.append(child)
            # 变异
            if random.random() < mutation_rate:
                ind_to_mutate = random.choice(new_population)
                pos_to_mutate = random.randint(0, 19)

                available_candidates = [c for c in candidates if c not in ind_to_mutate]
                if available_candidates:
                    new_num = random.choice(available_candidates)
                    ind_to_mutate[pos_to_mutate] = new_num
                    ind_to_mutate.sort()

        population = new_population

    final_fitness = [fitness(ind) for ind in population]
    best_individual = population[int(np.argmax(final_fitness))]
    return best_individual

def export_next_period_forecasts(
    predictor: TimeSeriesPredictor,
    full_df: pd.DataFrame,
    original_df: pd.DataFrame,
    out_dir: Path,
    dynamic_constraints: Optional[Dict[str, float]],
    static_features: Optional[pd.DataFrame] = None
):
    """Export next-period probability forecasts and supporting analysis."""
    historical_patterns = analyze_historical_patterns(original_df)
    static_features_df = prepare_static_features_for_ag(static_features)
    tdf_full = TimeSeriesDataFrame.from_data_frame(
        full_df,
        id_column='item_id',
        timestamp_column='timestamp',
        static_features_df=static_features_df
    ).fill_missing_values(method="auto")
    present_known = [c for c in KNOWN_COVARIATES if c in full_df.columns]
    if present_known:
        future_index_df = predictor.make_future_data_frame(tdf_full)
        last_known_values = (
            full_df.sort_values('timestamp')
            .groupby('item_id')
            .last()
            .reset_index()[['item_id'] + present_known]
        )
        future_cov_df = future_index_df.merge(last_known_values, on='item_id', how='left')
        known_cov_tsd = TimeSeriesDataFrame.from_data_frame(
            future_cov_df,
            id_column='item_id',
            timestamp_column='timestamp'
        ).fill_missing_values(method="auto")
    else:
        known_cov_tsd = None

    # 计算下一期时间戳
    next_date = full_df['timestamp'].max() + pd.Timedelta(days=1)
    qra_at_date: Optional[pd.DataFrame] = None
    out_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件路径
    fp = out_dir / (
        f'prob_forecast_optimized_PRESETS_{PRESETS}_'
        f'PREDICTION_LENGTH_{PREDICTION_LENGTH}_{next_date:%Y%m%d}.md'
    )

    # 写入表头
    header = ['model', 'timestamp', 'quantile'] + [f'pos_{i}' for i in range(1, 21)]
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')

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
                qra_at_date = combined_df
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

    final_pred_df: pd.DataFrame
    if qra_at_date is not None:
        final_pred_df = qra_at_date

        logger.info("使用 QRA 组合预测生成最终号码")
    else:
        final_pred_df = best_at_date
        logger.info("使用 ensemble 平均预测生成最终号码")
    # 对预测结果执行平滑与截断
    final_pred_df = postprocess_predictions(final_pred_df)
    final_numbers = compute_advanced_final_numbers(
        final_pred_df, historical_patterns, original_df, dynamic_constraints
    )

    # 保存最终号码到 CSV
    final_fp = out_dir / f'final_prediction_{next_date:%Y%m%d}.csv'
    df_final = pd.DataFrame(
        {f'number_{i + 1}': [num] for i, num in enumerate(final_numbers)},
        index=[next_date.strftime('%Y-%m-%d')]
    )
    df_final.index.name = 'timestamp'
    df_final.to_csv(final_fp)
    logger.info(f"已保存最终预测 CSV → {final_fp}")

    # —— 导出分析报告 ——
    analysis_fp = out_dir / f'prediction_analysis_{next_date:%Y%m%d}.txt'
    with open(analysis_fp, 'w', encoding='utf-8') as f:
        f.write(f"KL-8 预测分析报告 - {next_date.strftime('%Y-%m-%d')}\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. 最终预测数字:\n")
        f.write(f"   {final_numbers}\n\n")

        low_count = sum(1 for n in final_numbers if 1 <= n <= 20)
        mid_count = sum(1 for n in final_numbers if 21 <= n <= 60)
        high_count = sum(1 for n in final_numbers if 61 <= n <= 80)
        odd_count = sum(1 for n in final_numbers if n % 2 == 1)
        sum_val = sum(final_numbers)
        sorted_nums = sorted(final_numbers)
        consecutive = sum(1 for i in range(len(sorted_nums) - 1) if sorted_nums[i+1] - sorted_nums[i] == 1)

        f.write("2. 数字分布分析:\n")
        f.write(f"   低段(1-20): {low_count} 个\n")
        f.write(f"   中段(21-60): {mid_count} 个\n")
        f.write(f"   高段(61-80): {high_count} 个\n")
        f.write(f"   奇数: {odd_count} 个, 偶数: {20 - odd_count} 个\n")
        f.write(f"   总和: {sum_val}\n")
        f.write(f"   连号数量: {consecutive}\n\n")

        # 2.1 动态约束预测值
        if dynamic_constraints:
            f.write("2.1 动态约束预测 (来自元模型):\n")
            for k, v in dynamic_constraints.items():
                f.write(f"   {k}: {v:.2f}\n")
            f.write("\n")

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
        min_q = str(min(QUANTILE_LEVELS))
        max_q = str(max(QUANTILE_LEVELS))
        if (
            static_features is not None and not static_features.empty and
            min_q in final_pred_df.columns and max_q in final_pred_df.columns
        ):
            diag_df = final_pred_df.copy()
            diag_df['spread'] = (diag_df[max_q] - diag_df[min_q]).astype(float)
            diag_df = diag_df.join(static_features, how='left')
            if diag_df['spread'].isnull().any():
                fallback = float(diag_df['spread'].median(skipna=True)) if not diag_df['spread'].dropna().empty else 0.0
                diag_df['spread'] = diag_df['spread'].fillna(fallback)
            top_confident = diag_df.nsmallest(5, 'spread')
            top_uncertain = diag_df.nlargest(5, 'spread')

            f.write("\n5. 位置置信度概览:\n")
            f.write("   最稳定位置 (spread 最小):\n")
            for idx, row in top_confident.iterrows():
                f.write(
                    f"      {idx}: mean={row['mean']:.2f}, spread={row['spread']:.2f}, "
                    f"hist_mean={row.get('hist_target_mean', 0):.2f}, autocorr={row.get('hist_autocorr_lag1', 0):.2f}\n"
                )
            f.write("   波动较大的位置 (spread 最大):\n")
            for idx, row in top_uncertain.iterrows():
                f.write(
                    f"      {idx}: mean={row['mean']:.2f}, spread={row['spread']:.2f}, "
                    f"hist_std={row.get('hist_target_std', 0):.2f}, trend={row.get('hist_target_trend', 0):.2f}\n"
                )

    logger.info(f"已保存分析报告 → {analysis_fp}")

def run_backtest(
        predictor: TimeSeriesPredictor,
        data_df: pd.DataFrame,
        out_dir: Path,
        num_windows: int = 3,
        static_features: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """Run rolling backtests and return evaluation scores per cutoff."""
    static_features_df = prepare_static_features_for_ag(static_features)
    tdf = TimeSeriesDataFrame.from_data_frame(
        data_df,
        id_column='item_id',
        timestamp_column='timestamp',
        static_features_df=static_features_df
    ).fill_missing_values(method="auto")
    window_scores: Dict[str, float] = {}
    for cutoff in range(-num_windows * PREDICTION_LENGTH, 0, PREDICTION_LENGTH):
        try:
            score_dict = predictor.evaluate(tdf, cutoff=cutoff)
            if isinstance(score_dict, dict):
                metric_name = getattr(predictor, 'eval_metric', None)
                if metric_name:
                    val = score_dict.get(str(metric_name), None)
                else:
                    val = score_dict.get('WQL', None)
                if val is None and len(score_dict) > 0:
                    val = next(iter(score_dict.values()))
            else:
                val = score_dict

            window_scores[str(cutoff)] = float(val) if val is not None else float('nan')
            logger.info(f"回测窗口 cutoff={cutoff}: {val:.4f}")
        except Exception as e:
            logger.warning(f"回测评估 cutoff={cutoff} 失败: {e}")
            window_scores[str(cutoff)] = float('nan')
    out_dir.mkdir(parents=True, exist_ok=True)
    bt_fp = out_dir / 'backtest_results.csv'
    with open(bt_fp, 'w', encoding='utf-8') as f:
        f.write('cutoff,score\n')
        for c, s in window_scores.items():
            f.write(f"{c},{s}\n")

    valid_scores = [s for s in window_scores.values() if not np.isnan(s)]
    if valid_scores:
        avg_score = float(np.mean(valid_scores))
        std_score = float(np.std(valid_scores))
        logger.info("回测得分统计 → 均值: %.4f, 标准差: %.4f", avg_score, std_score)
    else:
        logger.info("回测得分统计 → 均值: nan, 标准差: nan")

    logger.info("已保存回测结果 → %s", bt_fp)
    return window_scores

def main():
    """主执行函数"""
    logger.info("开始KL-8深度优化预测流程")

    logger.info(f"正在从 {DATA_PATH} 加载数据...")
    try:
        raw = load_data(DATA_PATH)
    except FileNotFoundError as e:
        logger.error(f"无法加载数据，流程中止。错误: {e}")
        return

    dynamic_constraints = train_and_predict_meta_features(raw)

    logger.info("正在进行数据预处理...")
    df, static_features = preprocess_data(raw)

    logger.info("开始训练模型...")
    predictor = train_model(df, static_features)

    next_period_dir_name = (df['timestamp'].max() + pd.Timedelta(days=1)).strftime('%Y%m%d')
    out_dir = RESULTS_ROOT / next_period_dir_name

    logger.info("正在执行回测评估...")
    run_backtest(predictor, df, out_dir, num_windows=NUM_VAL_WINDOWS, static_features=static_features)

    logger.info("正在生成预测结果...")
    export_next_period_forecasts(predictor, df, raw, out_dir, dynamic_constraints, static_features)

    logger.info("KL-8深度优化预测流程完成！")

if __name__ == "__main__":
    main()
