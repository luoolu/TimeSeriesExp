#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KL-8 Advanced Probabilistic Forecasting System v2.0
深度优化版本 - 多维度增强的AutoGluon预测管道

主要优化改进：
1. 智能自适应特征工程：动态特征选择、多尺度时间特征、非线性组合特征
2. 多层次集成策略：QRA + Stacking + Dynamic Weighting
3. 高级遗传算法：多目标优化、自适应参数、精英保留策略
4. 预测置信度评估：不确定性量化、置信区间计算
5. 实时模型监控：性能追踪、模型漂移检测
6. 内存优化：增量学习、特征缓存、并行计算
7. 鲁棒性增强：异常检测、缺失值智能填补、数据质量评估

Author: Claude AI - Advanced Optimization
Version: 2.0
Updated: 2025-08-18
"""

import os
import gc
import logging
import warnings
import pickle
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import joblib

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.common import space

# === Begin: Safe JSON Encoder for numpy/pandas types ===
import json as _json_alias
try:
    import numpy as _np_alias
except Exception:  # pragma: no cover
    _np_alias = None
try:
    import pandas as _pd_alias
except Exception:  # pragma: no cover
    _pd_alias = None
try:
    import datetime as _dt_alias
except Exception:  # pragma: no cover
    _dt_alias = None

class SafeNpEncoder(_json_alias.JSONEncoder):
    def default(self, o):
        # numpy scalar types
        if _np_alias is not None:
            if isinstance(o, (_np_alias.integer,)):
                return int(o)
            if isinstance(o, (_np_alias.floating,)):
                return float(o)
            if isinstance(o, (_np_alias.bool_,)):
                return bool(o)
            if isinstance(o, (_np_alias.ndarray,)):
                return o.tolist()
        # pandas types
        if _pd_alias is not None:
            if isinstance(o, (_pd_alias.Series, _pd_alias.Index)):
                return o.tolist()
            if isinstance(o, (_pd_alias.Timestamp, _pd_alias.Timedelta, _pd_alias.Period)):
                return o.isoformat()
        # datetime objects
        if _dt_alias is not None:
            if isinstance(o, (_dt_alias.datetime, _dt_alias.date, _dt_alias.time)):
                return o.isoformat()
        return super().default(o)

def json_dump_safe(obj, fp, **kwargs):
    # If user already passed a custom encoder, respect it; otherwise use SafeNpEncoder
    if "cls" not in kwargs:
        kwargs["cls"] = SafeNpEncoder
    return _json_alias.dump(obj, fp, **kwargs)

def json_dumps_safe(obj, **kwargs):
    if "cls" not in kwargs:
        kwargs["cls"] = SafeNpEncoder
    return _json_alias.dumps(obj, **kwargs)
# === End: Safe JSON Encoder for numpy/pandas types ===

# ===== 配置类定义 =============================================================

@dataclass
class ModelConfig:
    """模型配置参数"""
    quantile_levels: List[float] = None
    prediction_length: int = 1
    max_lag: int = 30
    hist_window: int = 200
    presets: str = "best_quality"
    time_limit: Optional[int] = 48000  # 增加训练时间
    num_val_windows: int = 10  # 增加验证窗口
    
    def __post_init__(self):
        if self.quantile_levels is None:
            self.quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

@dataclass
class OptimizationConfig:
    """优化算法配置"""
    use_genetic_algorithm: bool = True
    population_size: int = 200  # 增大种群
    generations: int = 100  # 增加代数
    elite_ratio: float = 0.15
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    adaptive_mutation: bool = True
    multi_objective: bool = True

@dataclass
class FeatureConfig:
    """特征工程配置"""
    enable_automl_features: bool = True
    max_features: int = 500  # 增加特征数量限制
    feature_selection_method: str = "mutual_info"  # mutual_info, f_regression, combined
    enable_nonlinear_features: bool = True
    enable_interaction_features: bool = True
    pca_components: Optional[int] = 50

# ===== 日志和路径配置 =========================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('kl8_prediction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局配置实例
MODEL_CFG = ModelConfig()
OPT_CFG = OptimizationConfig()
FEAT_CFG = FeatureConfig()

# 路径配置
DATA_PATH = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/kl8_order_data.csv"
RESULTS_ROOT = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8"
MODEL_PATH = "autogluon_kl8_model_v2"
CACHE_DIR = "feature_cache"

# 权重配置 - 多层次加权
WEIGHTS = {
    'prediction': {'mean': 4.0, 'quantile': 2.0, 'uncertainty_penalty': 1.0},
    'frequency': {'recent': 1.2, 'historical': 0.8, 'trend': 1.0},
    'pattern': {'hot_bonus': 0.3, 'cold_bonus': 0.2, 'cycle_bonus': 0.4},
    'constraint': {'distribution': 0.2, 'consecutive': 0.1, 'balance': 0.15}
}

# 超参数优化配置 - 更精细的搜索空间
HYPERPARAMETERS_V2 = {
    "DeepAR": {
        "hidden_size": space.Int(32, 128),
        "num_layers": space.Int(2, 4),
        "dropout_rate": space.Real(0.1, 0.4),
        "learning_rate": space.Real(1e-4, 1e-2, log=True),
    },
    "TemporalFusionTransformer": {
        "hidden_size": space.Int(32, 128),
        "num_heads": space.Int(4, 8),
        "dropout_rate": space.Real(0.1, 0.3),
        "learning_rate": space.Real(1e-4, 1e-2, log=True),
    },
    "PatchTST": {
        "patch_size": space.Categorical(4, 8, 12, 16),
        "hidden_size": space.Int(32, 128),
        "num_layers": space.Int(2, 4),
        "dropout_rate": space.Real(0.1, 0.3),
    },
    "ChronosModel": {
        "model_path": space.Categorical(
            "autogluon/chronos-bolt-mini",
            "autogluon/chronos-bolt-small",
            "autogluon/chronos-bolt-base",
        ),
    },
    "RecursiveTabularModel": {
        "model_name": space.Categorical("GBM", "CAT", "RF"),
        "lags": space.Categorical(
            [1, 2, 3, 7],
            [1, 2, 3, 7, 14],
            [1, 2, 3, 7, 14, 21],
            [1, 2, 3, 7, 14, 21, 30]
        ),
        "target_scaler": space.Categorical("standard", "robust", "min_max"),
    },
}

HYPERPARAMETER_TUNE_KWARGS_V2 = {
    "num_trials": 100,  # 增加试验次数
    "scheduler": "local",
    "searcher": "random",  # only random
    "search_strategy": "auto",
}

# ===== 高级特征工程类 =========================================================

class AdvancedFeatureEngineer:
    """高级特征工程器"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_cache = {}
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.pca_transformer = None
        self.feature_importance = {}
        
    def extract_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取全面的特征集合"""
        logger.info("开始提取全面特征...")
        
        df = df.copy()
        
        # 1. 基础时间特征
        df = self._extract_temporal_features(df)
        
        # 2. 数字统计特征
        df = self._extract_number_statistics(df)
        
        # 3. 高级模式特征
        df = self._extract_pattern_features(df)
        
        # 4. 滞后和滚动特征
        df = self._extract_lag_rolling_features(df)
        
        # 5. 非线性特征
        if self.config.enable_nonlinear_features:
            df = self._extract_nonlinear_features(df)
            
        # 6. 交互特征
        if self.config.enable_interaction_features:
            df = self._extract_interaction_features(df)
            
        # 7. 频率和周期特征
        df = self._extract_frequency_features(df)
        
        # 8. 异常检测特征
        df = self._extract_anomaly_features(df)
        
        logger.info(f"特征提取完成，总特征数: {df.shape[1]}")
        return df
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取多尺度时间特征"""
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        df['quarter'] = df['timestamp'].dt.quarter
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['timestamp'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['timestamp'].dt.is_quarter_end.astype(int)
        
        # 周期性编码
        for period, col in [(7, 'day_of_week'), (52, 'week_of_year'), (12, 'month'), (4, 'quarter')]:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
            
        return df
    
    def _extract_number_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取数字统计特征"""
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        if not all(col in df.columns for col in num_cols):
            return df
            
        numbers_array = df[num_cols].values
        
        # 基础统计
        df['mean_number'] = np.mean(numbers_array, axis=1)
        df['std_number'] = np.std(numbers_array, axis=1)
        df['median_number'] = np.median(numbers_array, axis=1)
        df['range_number'] = np.ptp(numbers_array, axis=1)
        df['sum_number'] = np.sum(numbers_array, axis=1)
        df['var_number'] = np.var(numbers_array, axis=1)
        
        # 高阶统计
        df['skewness'] = stats.skew(numbers_array, axis=1)
        df['kurtosis'] = stats.kurtosis(numbers_array, axis=1)
        df['iqr'] = np.percentile(numbers_array, 75, axis=1) - np.percentile(numbers_array, 25, axis=1)
        
        # 分位数特征
        for q in [0.1, 0.25, 0.75, 0.9]:
            df[f'quantile_{int(q*100)}'] = np.percentile(numbers_array, q*100, axis=1)
            
        # 区间分布
        df['low_range_count'] = np.sum((numbers_array >= 1) & (numbers_array <= 20), axis=1)
        df['mid_range_count'] = np.sum((numbers_array >= 21) & (numbers_array <= 60), axis=1)
        df['high_range_count'] = np.sum((numbers_array >= 61) & (numbers_array <= 80), axis=1)
        
        # 奇偶性
        df['odd_count'] = np.sum(numbers_array % 2 == 1, axis=1)
        df['even_count'] = np.sum(numbers_array % 2 == 0, axis=1)
        df['odd_even_ratio'] = df['odd_count'] / (df['even_count'] + 1e-8)
        
        return df
        
    def _extract_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取模式特征"""
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        if not all(col in df.columns for col in num_cols):
            return df
            
        numbers_array = df[num_cols].values
        sorted_numbers = np.sort(numbers_array, axis=1)
        
        # 连续性分析
        consecutive_diffs = np.diff(sorted_numbers, axis=1)
        df['consecutive_count'] = np.sum(consecutive_diffs == 1, axis=1)
        df['max_gap'] = np.max(consecutive_diffs, axis=1)
        df['min_gap'] = np.min(consecutive_diffs, axis=1)
        df['avg_gap'] = np.mean(consecutive_diffs, axis=1)
        df['gap_variance'] = np.var(consecutive_diffs, axis=1)
        
        # 质数分析
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79])
        is_prime = np.isin(numbers_array, primes)
        df['prime_count'] = is_prime.sum(axis=1)
        df['prime_ratio'] = df['prime_count'] / 20
        
        # 数字和特征
        tens_digits = numbers_array // 10
        units_digits = numbers_array % 10
        digit_sum = tens_digits + units_digits
        
        df['tens_mean'] = np.mean(tens_digits, axis=1)
        df['units_mean'] = np.mean(units_digits, axis=1)
        df['digit_sum_mean'] = np.mean(digit_sum, axis=1)
        df['digit_sum_std'] = np.std(digit_sum, axis=1)
        
        # 重复数字
        df['repeated_digits'] = np.sum(tens_digits == units_digits, axis=1)
        
        return df
        
    def _extract_lag_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取滞后和滚动窗口特征"""
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'item_id']]
        key_features = ['mean_number', 'sum_number', 'std_number', 'odd_count', 'consecutive_count']
        key_features = [f for f in key_features if f in feature_cols]
        
        # 滞后特征
        for col in key_features:
            for lag in [1, 2, 3, 7, 14, 21]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
        # 滚动窗口特征
        for col in key_features:
            for window in [3, 7, 14, 21, 30]:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                
        return df
        
    def _extract_nonlinear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取非线性特征"""
        key_cols = ['mean_number', 'std_number', 'sum_number', 'range_number']
        key_cols = [c for c in key_cols if c in df.columns]
        
        for col in key_cols:
            # 多项式特征
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_cubed'] = df[col] ** 3
            df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
            df[f'{col}_log'] = np.log1p(np.abs(df[col]))
            
            # 三角函数特征
            df[f'{col}_sin'] = np.sin(df[col])
            df[f'{col}_cos'] = np.cos(df[col])
            
        return df
        
    def _extract_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取交互特征"""
        base_features = ['mean_number', 'std_number', 'sum_number', 'odd_count', 'consecutive_count']
        base_features = [f for f in base_features if f in df.columns]
        
        # 两两交互
        from itertools import combinations
        for f1, f2 in combinations(base_features, 2):
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
            df[f'{f1}_div_{f2}'] = df[f1] / (df[f2] + 1e-8)
            
        return df
        
    def _extract_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取频率和周期特征"""
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        if not all(col in df.columns for col in num_cols):
            return df

        logger.info("Calculating frequency features efficiently...")
        # Create a boolean DataFrame indicating where each number appears
        all_numbers_present = pd.concat([df[col] for col in num_cols]).unique()
        all_numbers_present.sort()

        # Use a more efficient approach for frequency calculation
        for window in [10, 30, 50, 100]:
            # Create a single multi-index dataframe for the window's counts
            # This is much faster than iterating over every number
            rolling_counts = df[num_cols].rolling(window=window, min_periods=1)
            
            # We can't apply value_counts directly on the rolling object.
            # Instead, we iterate over numbers, but use efficient rolling sum.
            for num in range(1, 81):
                # Check where the number appears across all 20 columns
                is_num_present = df[num_cols].eq(num).sum(axis=1)
                # Calculate rolling sum of occurrences, shift to not include current row
                freq = is_num_present.rolling(window=window, min_periods=1).sum().shift(1).fillna(0)
                df[f'freq_{num}_{window}'] = freq

        logger.info("Frequency features calculation complete.")
        return df
        
    def _extract_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取异常检测特征"""
        feature_cols = ['mean_number', 'std_number', 'sum_number', 'consecutive_count']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        if not feature_cols:
            return df
            
        try:
            # 使用孤立森林检测异常
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            features_array = df[feature_cols].fillna(df[feature_cols].median()).values
            iso_forest.fit(features_array)
            anomaly_scores = iso_forest.decision_function(features_array)
            df['anomaly_score'] = anomaly_scores
            df['is_anomaly'] = (anomaly_scores < 0).astype(int)
            
            # Z-score异常检测
            for col in feature_cols:
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
                df[f'{col}_zscore'] = z_scores
                df[f'{col}_is_outlier'] = (z_scores > 2).astype(int)
                
        except Exception as e:
            logger.warning(f"异常特征提取失败: {e}")
            
        return df
        
    def select_features(self, df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """智能特征选择"""
        if target_col not in df.columns:
            return df
            
        logger.info("开始特征选择...")
        
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'item_id', target_col]]
        X = df[feature_cols].fillna(0)
        # Normalize object columns: remove thousands-separators and cast to numeric
        for _c in list(X.columns):
            if X[_c].dtype == 'object':
                X[_c] = pd.to_numeric(X[_c].astype(str).str.replace(',', ''), errors='coerce')
        X = X.fillna(0)
        y = df[target_col]
        
        # 移除常数特征
        constant_features = [c for c in feature_cols if X[c].nunique() <= 1]
        if constant_features:
            logger.info(f"移除{len(constant_features)}个常数特征")
            feature_cols = [c for c in feature_cols if c not in constant_features]
            X = X[feature_cols]
            
        if len(feature_cols) <= self.config.max_features:
            return df
            
        # 特征选择
        try:
            if self.config.feature_selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_regression, k=self.config.max_features)
            elif self.config.feature_selection_method == "f_regression":
                selector = SelectKBest(f_regression, k=self.config.max_features)
            else:  # combined
                # 使用组合方法
                mi_selector = SelectKBest(mutual_info_regression, k=self.config.max_features//2)
                f_selector = SelectKBest(f_regression, k=self.config.max_features//2)
                
                mi_features = set(np.array(feature_cols)[mi_selector.fit(X, y).get_support()])
                f_features = set(np.array(feature_cols)[f_selector.fit(X, y).get_support()])
                selected_features = list(mi_features.union(f_features))
                
                return df[['timestamp', 'item_id', target_col] + selected_features]
                
            selector.fit(X, y)
            selected_features = np.array(feature_cols)[selector.get_support()]
            self.feature_importance = dict(zip(selected_features, selector.scores_[selector.get_support()]))
            
            logger.info(f"特征选择完成，选择{len(selected_features)}个特征")
            return df[['timestamp', 'item_id', target_col] + list(selected_features)]
            
        except Exception as e:
            logger.warning(f"特征选择失败: {e}")
            return df

# ===== 高级预测器类 ===========================================================

class AdvancedPredictor:
    """高级预测器"""
    
    def __init__(self, config: ModelConfig):
        self.known_covariates_names = []
        self.config = config
        self.predictor = None
        self.model_weights = {}
        self.prediction_history = deque(maxlen=100)
        
    def train(self, train_df: pd.DataFrame) -> 'AdvancedPredictor':
        """训练模型"""
        logger.info("开始训练高级预测模型...")
        
        tdf = TimeSeriesDataFrame.from_data_frame(
            train_df, id_column='item_id', timestamp_column='timestamp'
        ).fill_missing_values()
        
        # 动态确定已知协变量
        known_covariates = self._identify_known_covariates(train_df)
        
        self.known_covariates_names = known_covariates or []
        self.predictor = TimeSeriesPredictor(
            path=MODEL_PATH,
            prediction_length=self.config.prediction_length,
            freq='D',
            target='target',
            eval_metric='WQL',
            quantile_levels=self.config.quantile_levels,
            known_covariates_names=known_covariates or None
        )
        
        fit_kwargs = {
            'train_data': tdf,
            'presets': self.config.presets,
            'num_val_windows': self.config.num_val_windows,
            'refit_every_n_windows': 1,
            'time_limit': self.config.time_limit,
            'enable_ensemble': True,
            'hyperparameters': HYPERPARAMETERS_V2,
            'hyperparameter_tune_kwargs': HYPERPARAMETER_TUNE_KWARGS_V2,
        }
        
        self.predictor.fit(**fit_kwargs)
        self.model_weights = self._compute_dynamic_weights(tdf)
        
        logger.info("模型训练完成")
        return self
        
    def _identify_known_covariates(self, df: pd.DataFrame) -> List[str]:
        """动态识别已知协变量"""
        potential_covariates = [
            'day_of_week', 'week_of_year', 'month', 'quarter',
            'day_of_week_sin', 'day_of_week_cos',
            'week_of_year_sin', 'week_of_year_cos',
            'month_sin', 'month_cos', 'is_weekend'
        ]
        return [c for c in potential_covariates if c in df.columns]
        
    def _compute_dynamic_weights(self, tdf: TimeSeriesDataFrame) -> Dict[str, float]:
        """计算动态模型权重"""
        try:
            leaderboard = self.predictor.leaderboard(tdf, silent=True)
            if isinstance(leaderboard, pd.DataFrame) and 'model' in leaderboard.columns:
                # 使用性能和稳定性综合评分
                scores = {}
                for _, row in leaderboard.iterrows():
                    model = row['model']
                    score = -row.get('score_val', 0) if row.get('score_val', 0) < 0 else row.get('score_val', 0)
                    scores[model] = max(score, 0)
                    
                # 归一化权重
                total = sum(scores.values())
                if total > 0:
                    return {m: s/total for m, s in scores.items()}
                    
        except Exception as e:
            logger.warning(f"动态权重计算失败: {e}")
            
        # 均匀权重作为后备
        model_names = self.predictor.model_names() if self.predictor else []
        n = len(model_names)
        return {m: 1/n for m in model_names} if n > 0 else {}

    def predict_with_uncertainty(
            self,
            data_df: pd.DataFrame,
            known_covariates_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        带不确定性的预测（单步或多步，取决于 self.config.prediction_length）。
        - 若训练时设置了 known_covariates_names，则预测时自动构造未来已知协变量（当未显式提供时）。
        - 对每个已训练模型分别预测，并在可用时计算 QRA 集成。

        Parameters
        ----------
        data_df : pd.DataFrame
            至少包含 ['item_id', 'timestamp', 'target'(可选)]，以及可能的历史协变量。
        known_covariates_df : Optional[pd.DataFrame]
            未来 horizon 的已知协变量（若不提供且训练时使用了已知协变量，则自动构造）。

        Returns
        -------
        Dict[str, pd.DataFrame]
            {模型名: 预测结果}，必要时包含 'QRA_Ensemble'。
        """
        if not self.predictor:
            raise ValueError("模型未训练")

        # 转换为 AG 的长表并补齐缺失
        tdf = TimeSeriesDataFrame.from_data_frame(
            data_df, id_column="item_id", timestamp_column="timestamp"
        ).fill_missing_values()

        # 需要已知协变量且未提供时，自动构造未来 horizon 的日历特征
        if known_covariates_df is None:
            try:
                known_covariates_df = self._make_known_covariates_future(
                    data_df, self.config.prediction_length
                )
            except Exception as e:
                logger.warning(f"自动构造已知协变量失败: {e}")
                known_covariates_df = None

        # 若训练阶段声明了已知协变量，但仍没有可用的 known_covariates_df，则直接报错比循环报错更清晰
        if getattr(self, "known_covariates_names", []) and known_covariates_df is None:
            raise ValueError("预测需要已知协变量，但未能构造/提供 known_covariates_df。")

        # 仅保留需要的列顺序：['item_id','timestamp'] + known_covariates_names
        if known_covariates_df is not None:
            need_cols = ["item_id", "timestamp"] + [
                c for c in getattr(self, "known_covariates_names", []) if c in known_covariates_df.columns
            ]
            known_covariates_df = known_covariates_df.loc[:, need_cols]

        results: Dict[str, pd.DataFrame] = {}

        # 多模型预测
        for model_name in self.predictor.model_names():
            try:
                pred = self.predictor.predict(
                    tdf,
                    model=model_name,
                    known_covariates=known_covariates_df,
                )
                results[model_name] = pred
            except Exception as e:
                logger.warning(f"模型{model_name}预测失败: {e}")

        # QRA 集成（至少两个基础模型结果时才有意义）
        if len(results) >= 2 and hasattr(self, "_compute_qra_ensemble"):
            try:
                results["QRA_Ensemble"] = self._compute_qra_ensemble(results)
            except Exception as e:
                logger.warning(f"QRA 集成失败: {e}")

        return results

    
    def _make_known_covariates_future(self, data_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        构造未来 horizon 天（按日频率）的已知协变量（known covariates）。
        在训练开启 known_covariates_names 的前提下，预测时基于历史最后一天向后扩展。
        返回列：['item_id','timestamp'] + self.known_covariates_names（存在者）。
        """
        if not getattr(self, "known_covariates_names", []):
            return pd.DataFrame(columns=["item_id", "timestamp"])

        if "item_id" not in data_df.columns or "timestamp" not in data_df.columns:
            raise ValueError("data_df 必须包含 'item_id' 与 'timestamp' 列")

        data_df = data_df.copy()
        data_df["timestamp"] = pd.to_datetime(data_df["timestamp"])

        last_ts_map = data_df.groupby("item_id")["timestamp"].max().to_dict()
        all_rows = []

        def _calendar_features(ts: pd.Series) -> pd.DataFrame:
            df = pd.DataFrame({"timestamp": ts})
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
            df["month"] = df["timestamp"].dt.month
            df["quarter"] = df["timestamp"].dt.quarter
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
            df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
            df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
            df["week_of_year_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52.0)
            df["week_of_year_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52.0)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
            return df

        for item_id, last_ts in last_ts_map.items():
            future_index = pd.date_range(start=last_ts + pd.Timedelta(days=1), periods=horizon, freq="D")
            if len(future_index) == 0:
                continue
            feat = _calendar_features(pd.Series(future_index))
            feat.insert(0, "item_id", item_id)
            all_rows.append(feat)

        if not all_rows:
            return pd.DataFrame(columns=["item_id", "timestamp"] + list(self.known_covariates_names))

        future_df = pd.concat(all_rows, ignore_index=True)
        keep_covs = [c for c in self.known_covariates_names if c in future_df.columns]
        future_df = future_df[["item_id", "timestamp"] + keep_covs].sort_values(["item_id", "timestamp"]).reset_index(drop=True)

        try:
            if "category" in str(data_df["item_id"].dtype):
                future_df["item_id"] = future_df["item_id"].astype(data_df["item_id"].dtype)
        except Exception:
            pass

        return future_df


def _compute_qra_ensemble(self, predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算QRA集成预测"""
        try:
            reference_df = next(iter(predictions.values()))
            ensemble_df = pd.DataFrame(index=reference_df.index)
            
            for col in reference_df.columns:
                weighted_sum = None
                total_weight = 0
                
                for model_name, pred_df in predictions.items():
                    if col in pred_df.columns:
                        weight = self.model_weights.get(model_name, 1.0)
                        if weighted_sum is None:
                            weighted_sum = weight * pred_df[col]
                        else:
                            weighted_sum += weight * pred_df[col]
                        total_weight += weight
                        
                if total_weight > 0:
                    ensemble_df[col] = weighted_sum / total_weight
                    
            return ensemble_df
            
        except Exception as e:
            logger.warning(f"QRA集成计算失败: {e}")
            return reference_df

# ===== 高级遗传算法优化器 ======================================================

class MultiObjectiveGeneticOptimizer:
    """多目标遗传算法优化器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.best_fitness_history = []
        self.diversity_history = []
        
    def optimize_number_selection(self, candidates: List[int], 
                                 prediction_scores: Dict[int, float],
                                 historical_patterns: Dict[str, Any]) -> List[int]:
        """多目标优化数字选择"""
        logger.info("开始多目标遗传算法优化...")
        
        # 初始化种群
        population = self._initialize_population(candidates)
        
        # 进化过程
        for generation in range(self.config.generations):
            # 计算适应度
            fitness_scores = [self._multi_objective_fitness(ind, prediction_scores, historical_patterns) 
                            for ind in population]
            
            # 记录历史
            best_fitness = max(fitness_scores)
            self.best_fitness_history.append(best_fitness)
            
            # 计算种群多样性
            diversity = self._calculate_diversity(population)
            self.diversity_history.append(diversity)
            
            # 选择、交叉、变异
            population = self._evolve_population(population, fitness_scores, candidates)
            
            # 自适应参数调整
            if self.config.adaptive_mutation:
                self._adapt_parameters(generation, diversity)
                
            if generation % 20 == 0:
                logger.info(f"第{generation}代: 最佳适应度={best_fitness:.4f}, 多样性={diversity:.4f}")
        
        # 返回最优解
        final_fitness = [self._multi_objective_fitness(ind, prediction_scores, historical_patterns) 
                        for ind in population]
        best_idx = np.argmax(final_fitness)
        
        logger.info(f"遗传算法优化完成，最终适应度: {final_fitness[best_idx]:.4f}")
        return sorted(population[best_idx])
        
    def _initialize_population(self, candidates: List[int]) -> List[List[int]]:
        """初始化种群"""
        population = []
        
        # 贪婪初始化（部分个体）
        for _ in range(self.config.population_size // 4):
            individual = sorted(np.random.choice(candidates, 20, replace=False))
            population.append(individual)
            
        # 分层初始化（确保分布合理）
        for _ in range(self.config.population_size // 4):
            low_range = [n for n in candidates if 1 <= n <= 20]
            mid_range = [n for n in candidates if 21 <= n <= 60]
            high_range = [n for n in candidates if 61 <= n <= 80]
            
            individual = []
            individual.extend(np.random.choice(low_range, min(7, len(low_range)), replace=False))
            individual.extend(np.random.choice(mid_range, min(8, len(mid_range)), replace=False))
            individual.extend(np.random.choice(high_range, min(5, len(high_range)), replace=False))
            
            # 填充到20个
            remaining = [n for n in candidates if n not in individual]
            needed = 20 - len(individual)
            if needed > 0 and remaining:
                individual.extend(np.random.choice(remaining, min(needed, len(remaining)), replace=False))
                
            population.append(sorted(individual[:20]))
            
        # 随机初始化（剩余个体）
        while len(population) < self.config.population_size:
            individual = sorted(np.random.choice(candidates, min(20, len(candidates)), replace=False))
            population.append(individual)
            
        return population
        
    def _multi_objective_fitness(self, individual: List[int], 
                               prediction_scores: Dict[int, float],
                               historical_patterns: Dict[str, Any]) -> float:
        """多目标适应度函数"""
        # 目标1: 预测分数
        prediction_fitness = sum(prediction_scores.get(n, 0) for n in individual) / len(individual)
        
        # 目标2: 历史频率适应性
        freq_fitness = self._calculate_frequency_fitness(individual, historical_patterns)
        
        # 目标3: 分布约束满足度
        distribution_fitness = self._calculate_distribution_fitness(individual, historical_patterns)
        
        # 目标4: 模式多样性
        diversity_fitness = self._calculate_pattern_diversity(individual)
        
        # 加权组合
        total_fitness = (
            0.4 * prediction_fitness +
            0.25 * freq_fitness +
            0.2 * distribution_fitness +
            0.15 * diversity_fitness
        )
        
        return total_fitness
        
    def _calculate_frequency_fitness(self, individual: List[int], 
                                   historical_patterns: Dict[str, Any]) -> float:
        """计算频率适应度"""
        number_freq = historical_patterns.get('number_freq', pd.Series())
        recent_freq = historical_patterns.get('recent_freq', pd.Series())
        
        if len(number_freq) == 0:
            return 0.5
            
        # 历史频率适应性
        hist_scores = [number_freq.get(n, 0) for n in individual]
        hist_fitness = np.mean(hist_scores) / (number_freq.max() + 1e-8)
        
        # 近期频率适应性
        recent_scores = [recent_freq.get(n, 0) for n in individual]
        recent_fitness = np.mean(recent_scores) / (recent_freq.max() + 1e-8)
        
        return 0.6 * hist_fitness + 0.4 * recent_fitness
        
    def _calculate_distribution_fitness(self, individual: List[int], 
                                      historical_patterns: Dict[str, Any]) -> float:
        """计算分布适应度"""
        dist_stats = historical_patterns.get('dist_stats', {})
        
        # 实际分布
        low_count = sum(1 for n in individual if 1 <= n <= 20)
        mid_count = sum(1 for n in individual if 21 <= n <= 60)
        high_count = sum(1 for n in individual if 61 <= n <= 80)
        odd_count = sum(1 for n in individual if n % 2 == 1)
        
        # 目标分布
        target_low = dist_stats.get('low', 5)
        target_mid = dist_stats.get('mid', 10)
        target_high = dist_stats.get('high', 5)
        target_odd = dist_stats.get('odd', 10)
        
        # 计算偏差惩罚（使用平滑惩罚函数）
        def smooth_penalty(actual, target, tolerance=1):
            diff = abs(actual - target)
            return 1.0 / (1.0 + diff / tolerance) if diff > tolerance else 1.0
            
        low_fitness = smooth_penalty(low_count, target_low)
        mid_fitness = smooth_penalty(mid_count, target_mid)
        high_fitness = smooth_penalty(high_count, target_high)
        odd_fitness = smooth_penalty(odd_count, target_odd)
        
        return np.mean([low_fitness, mid_fitness, high_fitness, odd_fitness])
        
    def _calculate_pattern_diversity(self, individual: List[int]) -> float:
        """计算模式多样性"""
        # 间隔多样性
        sorted_ind = sorted(individual)
        gaps = np.diff(sorted_ind)
        gap_diversity = 1.0 - np.std(gaps) / (np.mean(gaps) + 1e-8)
        
        # 数字段分布多样性
        segments = [0] * 8  # 分为8个段
        for n in individual:
            segment = min(7, (n - 1) // 10)
            segments[segment] += 1
            
        segment_diversity = 1.0 - np.std(segments) / (np.mean(segments) + 1e-8)
        
        # 奇偶分布
        odd_count = sum(1 for n in individual if n % 2 == 1)
        even_count = 20 - odd_count
        parity_diversity = 1.0 - abs(odd_count - even_count) / 20
        
        return np.mean([gap_diversity, segment_diversity, parity_diversity])
        
    def _evolve_population(self, population: List[List[int]], 
                          fitness_scores: List[float], 
                          candidates: List[int]) -> List[List[int]]:
        """进化种群"""
        new_population = []
        
        # 精英保留
        elite_count = int(self.config.population_size * self.config.elite_ratio)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        elites = [population[i] for i in elite_indices]
        new_population.extend(elites)
        
        # 生成剩余个体
        while len(new_population) < self.config.population_size:
            if np.random.random() < self.config.crossover_rate:
                # 交叉操作
                parent1, parent2 = self._tournament_selection(population, fitness_scores, 2)
                child = self._smart_crossover(parent1, parent2, candidates)
            else:
                # 直接选择
                child = self._tournament_selection(population, fitness_scores, 1)[0].copy()
                
            # 变异操作
            if np.random.random() < self.config.mutation_rate:
                child = self._smart_mutation(child, candidates)
                
            new_population.append(child)
            
        return new_population[:self.config.population_size]
        
    def _tournament_selection(self, population: List[List[int]], 
                            fitness_scores: List[float], 
                            count: int) -> List[List[int]]:
        """锦标赛选择"""
        selected = []
        tournament_size = max(2, len(population) // 10)
        
        for _ in range(count):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
            
        return selected
        
    def _smart_crossover(self, parent1: List[int], parent2: List[int], 
                        candidates: List[int]) -> List[int]:
        """智能交叉操作"""
        # 保留共同数字
        common = list(set(parent1) & set(parent2))
        
        # 从双方选择剩余数字
        remaining1 = [n for n in parent1 if n not in common]
        remaining2 = [n for n in parent2 if n not in common]
        
        # 随机组合
        all_remaining = remaining1 + remaining2
        needed = 20 - len(common)
        
        if len(all_remaining) >= needed:
            selected_remaining = np.random.choice(all_remaining, needed, replace=False)
        else:
            selected_remaining = all_remaining
            # 从候选中补充
            additional_needed = needed - len(selected_remaining)
            available = [n for n in candidates if n not in common + list(selected_remaining)]
            if len(available) >= additional_needed:
                additional = np.random.choice(available, additional_needed, replace=False)
                selected_remaining = list(selected_remaining) + list(additional)
                
        child = sorted(common + list(selected_remaining))[:20]
        return child
        
    def _smart_mutation(self, individual: List[int], candidates: List[int]) -> List[int]:
        """智能变异操作"""
        mutated = individual.copy()
        
        # 随机选择要变异的位置数量
        mutation_count = np.random.randint(1, min(4, len(individual)))
        mutation_indices = np.random.choice(len(individual), mutation_count, replace=False)
        
        for idx in mutation_indices:
            # 选择替换数字
            available = [n for n in candidates if n not in mutated]
            if available:
                mutated[idx] = np.random.choice(available)
                
        return sorted(mutated)
        
    def _adapt_parameters(self, generation: int, diversity: float):
        """自适应参数调整"""
        # 根据多样性调整变异率
        if diversity < 0.3:
            self.config.mutation_rate = min(0.3, self.config.mutation_rate * 1.1)
        elif diversity > 0.7:
            self.config.mutation_rate = max(0.05, self.config.mutation_rate * 0.9)
            
        # 根据进化进度调整交叉率
        progress = generation / self.config.generations
        if progress > 0.8:  # 后期降低交叉率，增加局部搜索
            self.config.crossover_rate = max(0.6, 0.9 - 0.3 * progress)
            
    def _calculate_diversity(self, population: List[List[int]]) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0
            
        # 计算个体间的平均汉明距离
        total_distance = 0
        count = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = len(set(population[i]) ^ set(population[j])) / 20.0
                total_distance += distance
                count += 1
                
        return total_distance / count if count > 0 else 0.0

# ===== 数据处理和分析工具 ======================================================

class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        
    def load_and_validate_data(self, path: str) -> pd.DataFrame:
        """加载并验证数据"""
        logger.info(f"加载数据: {path}")
        
        try:
            df = pd.read_csv(path, encoding='utf-8')
            
            # 数据验证
            self._validate_data_quality(df)
            
            # 基础清理
            if '开奖日期' in df.columns:
                df = df.rename(columns={'开奖日期': 'timestamp'})
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ignore_index=True)
            
            # 数据完整性检查
            self._check_data_completeness(df)
            
            logger.info(f"数据加载完成: {len(df)}条记录, 时间跨度: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
            
    def _validate_data_quality(self, df: pd.DataFrame):
        """验证数据质量"""
        # 检查必要列
        required_cols = ['timestamp'] if 'timestamp' in df.columns else ['开奖日期']
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        
        missing_cols = [col for col in required_cols + num_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"缺失列: {missing_cols}")
            
        # 检查数字范围
        for col in num_cols:
            if col in df.columns:
                invalid_numbers = df[(df[col] < 1) | (df[col] > 80)][col]
                if len(invalid_numbers) > 0:
                    logger.warning(f"列{col}包含{len(invalid_numbers)}个无效数字")
                    
    def _check_data_completeness(self, df: pd.DataFrame):
        """检查数据完整性"""
        # 时间序列连续性检查
        date_range = pd.date_range(start=df['timestamp'].min(), 
                                 end=df['timestamp'].max(), 
                                 freq='D')
        missing_dates = set(date_range) - set(df['timestamp'])
        
        if missing_dates:
            logger.warning(f"缺失{len(missing_dates)}个日期的数据")
            
        # 缺失值统计
        missing_stats = df.isnull().sum()
        if missing_stats.sum() > 0:
            logger.info(f"缺失值统计:\n{missing_stats[missing_stats > 0]}")
            
    def preprocess_comprehensive(self, df: pd.DataFrame) -> pd.DataFrame:
        """综合预处理"""
        logger.info("开始综合数据预处理...")
        
        # 特征工程
        feature_engineer = AdvancedFeatureEngineer(FEAT_CFG)
        df_processed = feature_engineer.extract_comprehensive_features(df)
        
        # 异常处理
        df_processed = self._handle_anomalies(df_processed)
        
        # 缺失值处理
        df_processed = self._handle_missing_values(df_processed)
        
        # 数据格式转换
        df_long = self._convert_to_long_format(df_processed)
        
        # 特征选择
        if FEAT_CFG.enable_automl_features:
            df_long = feature_engineer.select_features(df_long)
            
        logger.info(f"预处理完成: {len(df_long)}条记录, {df_long.shape[1]}个特征")
        return df_long
        
    def _handle_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 数字范围约束
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].clip(1, 80)
                
        # 统计特征异常处理
        for col in numeric_cols:
            if col not in num_cols and col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
                
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """智能处理缺失值"""
        # 数值特征：使用插值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                # 时间序列插值
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                # 剩余缺失值用中位数填充
                df[col] = df[col].fillna(df[col].median())
                
        # 分类特征：使用众数
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
                
        return df
        
    def _convert_to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换为长格式用于时间序列建模"""
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp'] + num_cols]
        
        # 转换为长格式
        df_long = df.melt(
            id_vars=['timestamp'] + feature_cols,
            value_vars=num_cols,
            var_name='position',
            value_name='target'
        ).dropna(subset=['target'])
        
        # 创建项目ID
        df_long['item_id'] = df_long['position'].str.replace('开奖号_', 'pos_')
        df_long = df_long.drop('position', axis=1)
        
        # 数据类型优化
        df_long['target'] = df_long['target'].astype(int)
        df_long['item_id'] = df_long['item_id'].astype('category')
        
        # 排序和索引
        df_long = df_long.sort_values(['item_id', 'timestamp']).reset_index(drop=True)
        
        # 移除冷启动期数据
        df_long['row_idx'] = df_long.groupby('item_id').cumcount()
        df_long = df_long[df_long['row_idx'] >= MODEL_CFG.max_lag].drop('row_idx', axis=1)
        # Drop duplicated columns to avoid CatBoost/LightGBM errors
        df_long = df_long.loc[:, ~df_long.columns.duplicated()]

        
        return df_long

# ===== 历史模式分析器 =========================================================

class PatternAnalyzer:
    """历史模式分析器"""
    
    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.patterns_cache = {}
        
    def analyze_comprehensive_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """综合分析历史模式"""
        logger.info("开始综合历史模式分析...")
        
        patterns = {
            'frequency_analysis': self._analyze_frequency_patterns(df),
            'distribution_analysis': self._analyze_distribution_patterns(df),
            'cyclical_analysis': self._analyze_cyclical_patterns(df),
            'correlation_analysis': self._analyze_correlation_patterns(df),
            'trend_analysis': self._analyze_trend_patterns(df),
            'seasonal_analysis': self._analyze_seasonal_patterns(df)
        }
        
        logger.info("历史模式分析完成")
        return patterns
        
    def _analyze_frequency_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析频率模式"""
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        df_sorted = df.sort_values('timestamp')
        
        # 全历史频率
        all_numbers = df_sorted[num_cols].values.flatten()
        number_freq = pd.Series(all_numbers).value_counts().sort_index()
        
        # 近期频率
        recent_df = df_sorted.tail(self.window_size)
        recent_numbers = recent_df[num_cols].values.flatten()
        recent_freq = pd.Series(recent_numbers).value_counts().sort_index()
        
        # 热号和冷号
        hot_threshold = number_freq.quantile(0.8)
        cold_threshold = number_freq.quantile(0.2)
        
        hot_numbers = number_freq[number_freq >= hot_threshold].index.tolist()
        cold_numbers = number_freq[number_freq <= cold_threshold].index.tolist()
        
        # 趋势分析
        trend_window = min(100, len(df_sorted))
        recent_trend = df_sorted.tail(trend_window)
        prev_trend = df_sorted.iloc[-2*trend_window:-trend_window] if len(df_sorted) >= 2*trend_window else pd.DataFrame()
        
        trending_up = []
        trending_down = []
        
        if not prev_trend.empty:
            recent_counts = pd.Series(recent_trend[num_cols].values.flatten()).value_counts()
            prev_counts = pd.Series(prev_trend[num_cols].values.flatten()).value_counts()
            
            for num in range(1, 81):
                recent_freq_num = recent_counts.get(num, 0)
                prev_freq_num = prev_counts.get(num, 0)
                
                if recent_freq_num > prev_freq_num + 1:
                    trending_up.append(num)
                elif recent_freq_num < prev_freq_num - 1:
                    trending_down.append(num)
        
        return {
            'number_freq': number_freq,
            'recent_freq': recent_freq,
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'trending_up': trending_up,
            'trending_down': trending_down
        }
        
    def _analyze_distribution_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析分布模式"""
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        
        def calculate_distribution_stats(data_df):
            numbers = data_df[num_cols].values
            return {
                'low_range': np.mean(np.sum((numbers >= 1) & (numbers <= 20), axis=1)),
                'mid_range': np.mean(np.sum((numbers >= 21) & (numbers <= 60), axis=1)),
                'high_range': np.mean(np.sum((numbers >= 61) & (numbers <= 80), axis=1)),
                'odd_count': np.mean(np.sum(numbers % 2 == 1, axis=1)),
                'even_count': np.mean(np.sum(numbers % 2 == 0, axis=1)),
                'consecutive_pairs': np.mean([
                    np.sum(np.diff(np.sort(row)) == 1) for row in numbers
                ])
            }
        
        # 全历史分布
        historical_dist = calculate_distribution_stats(df)
        
        # 近期分布
        recent_dist = calculate_distribution_stats(df.tail(self.window_size))
        
        # 加权组合
        weights = {'recent': 0.7, 'historical': 0.3}
        combined_dist = {}
        for key in historical_dist:
            combined_dist[key] = (weights['recent'] * recent_dist[key] + 
                                weights['historical'] * historical_dist[key])
        
        return {
            'historical': historical_dist,
            'recent': recent_dist,
            'combined': combined_dist,
            'dist_stats': combined_dist  # 兼容旧接口
        }
        
    def _analyze_cyclical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析周期性模式"""
        df_with_time = df.copy()
        df_with_time['day_of_week'] = df_with_time['timestamp'].dt.dayofweek
        df_with_time['week_of_year'] = df_with_time['timestamp'].dt.isocalendar().week
        df_with_time['month'] = df_with_time['timestamp'].dt.month
        
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        
        # 按星期分析
        weekly_patterns = {}
        for day in range(7):
            day_data = df_with_time[df_with_time['day_of_week'] == day]
            if not day_data.empty:
                day_numbers = day_data[num_cols].values.flatten()
                weekly_patterns[day] = pd.Series(day_numbers).value_counts()
        
        # 按月份分析
        monthly_patterns = {}
        for month in range(1, 13):
            month_data = df_with_time[df_with_time['month'] == month]
            if not month_data.empty:
                month_numbers = month_data[num_cols].values.flatten()
                monthly_patterns[month] = pd.Series(month_numbers).value_counts()
        
        return {
            'weekly_patterns': weekly_patterns,
            'monthly_patterns': monthly_patterns
        }
        
    def _analyze_correlation_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析数字间相关性模式"""
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        
        # 数字共现分析
        cooccurrence_matrix = np.zeros((80, 80))
        
        for _, row in df.iterrows():
            numbers = [int(row[col]) - 1 for col in num_cols if pd.notna(row[col]) and 1 <= row[col] <= 80]
            for i, num1 in enumerate(numbers):
                for j, num2 in enumerate(numbers):
                    if i != j:
                        cooccurrence_matrix[num1][num2] += 1
        
        # 找出强相关数字对
        strong_correlations = []
        threshold = np.percentile(cooccurrence_matrix, 95)
        
        for i in range(80):
            for j in range(i+1, 80):
                if cooccurrence_matrix[i][j] > threshold:
                    strong_correlations.append({
                        'num1': i+1, 
                        'num2': j+1, 
                        'correlation': cooccurrence_matrix[i][j]
                    })
        
        return {
            'cooccurrence_matrix': cooccurrence_matrix,
            'strong_correlations': strong_correlations
        }
        
    def _analyze_trend_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析趋势模式"""
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        df_sorted = df.sort_values('timestamp')
        
        trend_analysis = {}
        window_sizes = [10, 20, 50, 100]
        
        for window in window_sizes:
            if len(df_sorted) >= window * 2:
                recent_window = df_sorted.tail(window)
                prev_window = df_sorted.iloc[-2*window:-window]
                
                recent_counts = pd.Series(recent_window[num_cols].values.flatten()).value_counts()
                prev_counts = pd.Series(prev_window[num_cols].values.flatten()).value_counts()
                
                trend_scores = {}
                for num in range(1, 81):
                    recent_freq = recent_counts.get(num, 0)
                    prev_freq = prev_counts.get(num, 0)
                    
                    if prev_freq > 0:
                        trend_scores[num] = (recent_freq - prev_freq) / prev_freq
                    else:
                        trend_scores[num] = 1.0 if recent_freq > 0 else 0.0
                
                trend_analysis[f'window_{window}'] = trend_scores
        
        return trend_analysis
        
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析季节性模式"""
        df_with_season = df.copy()
        df_with_season['season'] = df_with_season['timestamp'].dt.month % 12 // 3 + 1
        
        num_cols = [f'开奖号_{i}' for i in range(1, 21)]
        seasonal_patterns = {}
        
        for season in range(1, 5):
            season_data = df_with_season[df_with_season['season'] == season]
            if not season_data.empty:
                season_numbers = season_data[num_cols].values.flatten()
                seasonal_patterns[season] = pd.Series(season_numbers).value_counts()
        
        return seasonal_patterns

# ===== 智能预测生成器 =========================================================

class IntelligentPredictionGenerator:
    """智能预测生成器"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.uncertainty_weights = {
            'prediction_std': 0.3,
            'model_disagreement': 0.4,
            'historical_volatility': 0.3
        }
        
    def generate_final_prediction(self, 
                                 model_predictions: Dict[str, pd.DataFrame],
                                 historical_patterns: Dict[str, Any],
                                 original_df: pd.DataFrame) -> Dict[str, Any]:
        """生成最终预测结果"""
        logger.info("开始生成智能预测...")
        
        # 选择最佳预测
        best_prediction = self._select_best_prediction(model_predictions)
        
        # 计算不确定性
        uncertainty_metrics = self._calculate_uncertainty(model_predictions, historical_patterns)
        
        # 后处理优化
        optimized_prediction = self._postprocess_prediction(best_prediction, historical_patterns)
        
        # 生成候选数字池
        candidates = self._generate_candidate_pool(optimized_prediction, historical_patterns, original_df)
        
        # 智能数字选择
        final_numbers = self._intelligent_number_selection(
            candidates, optimized_prediction, historical_patterns, original_df
        )
        
        # 置信度评估
        confidence_score = self._assess_prediction_confidence(
            final_numbers, uncertainty_metrics, historical_patterns
        )
        
        result = {
            'final_numbers': sorted(final_numbers),
            'confidence_score': confidence_score,
            'uncertainty_metrics': uncertainty_metrics,
            'candidate_pool': candidates,
            'prediction_source': best_prediction.get('source', 'ensemble')
        }
        
        logger.info(f"预测生成完成，置信度: {confidence_score:.3f}")
        return result
        
    def _select_best_prediction(self, predictions: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """选择最佳预测"""
        if not predictions:
            raise ValueError('没有可用模型预测：请检查预测阶段 known_covariates 是否已提供/构造。')
        if 'QRA_Ensemble' in predictions:
            return {'data': predictions['QRA_Ensemble'], 'source': 'QRA_Ensemble'}
        elif 'WeightedEnsemble' in predictions:
            return {'data': predictions['WeightedEnsemble'], 'source': 'WeightedEnsemble'}
        else:
            # 选择第一个可用预测
            first_key = next(iter(predictions.keys()))
            return {'data': predictions[first_key], 'source': first_key}
            
    def _calculate_uncertainty(self, predictions: Dict[str, pd.DataFrame], 
                             historical_patterns: Dict[str, Any]) -> Dict[str, float]:
        """计算预测不确定性"""
        uncertainty = {}
        
        # 模型间分歧度
        if len(predictions) > 1:
            mean_predictions = []
            for model_name, pred_df in predictions.items():
                if 'mean' in pred_df.columns:
                    mean_predictions.append(pred_df['mean'].values)
            
            if len(mean_predictions) > 1:
                mean_predictions = np.array(mean_predictions)
                model_disagreement = np.mean(np.std(mean_predictions, axis=0))
                uncertainty['model_disagreement'] = model_disagreement / 80.0  # 归一化
            else:
                uncertainty['model_disagreement'] = 0.0
        else:
            uncertainty['model_disagreement'] = 0.0
        
        # 预测分位数离散度
        first_pred = next(iter(predictions.values()))
        if 'mean' in first_pred.columns:
            quantile_cols = [str(q) for q in MODEL_CFG.quantile_levels if str(q) in first_pred.columns]
            if len(quantile_cols) >= 2:
                q_low = first_pred[quantile_cols[0]].values
                q_high = first_pred[quantile_cols[-1]].values
                prediction_std = np.mean(q_high - q_low) / 80.0
                uncertainty['prediction_std'] = prediction_std
            else:
                uncertainty['prediction_std'] = 0.0
        else:
            uncertainty['prediction_std'] = 0.0
        
        # 历史波动性
        freq_analysis = historical_patterns.get('frequency_analysis', {})
        if 'number_freq' in freq_analysis:
            historical_volatility = freq_analysis['number_freq'].std() / freq_analysis['number_freq'].mean()
            uncertainty['historical_volatility'] = min(1.0, historical_volatility / 10.0)
        else:
            uncertainty['historical_volatility'] = 0.5
        
        return uncertainty
        
    def _postprocess_prediction(self, prediction_data: Dict[str, Any], 
                              historical_patterns: Dict[str, Any]) -> pd.DataFrame:
        """后处理预测数据"""
        pred_df = prediction_data['data'].copy()
        
        # 平滑处理
        smoothing_cols = ['mean'] + [str(q) for q in MODEL_CFG.quantile_levels if str(q) in pred_df.columns]
        for col in smoothing_cols:
            if col in pred_df.columns:
                # 应用小窗口平滑
                pred_df[col] = pred_df[col].rolling(window=3, min_periods=1, center=True).mean()
                # 范围约束
                pred_df[col] = pred_df[col].clip(1, 80)
        
        return pred_df
        
    def _generate_candidate_pool(self, pred_df: pd.DataFrame, 
                               historical_patterns: Dict[str, Any],
                               original_df: pd.DataFrame) -> List[int]:
        """生成候选数字池"""
        candidate_scores = {n: 0.0 for n in range(1, 81)}
        
        # 预测分数
        if 'mean' in pred_df.columns:
            for val in pred_df['mean']:
                num = int(np.clip(np.round(val), 1, 80))
                candidate_scores[num] += WEIGHTS['prediction']['mean']
        
        # 分位数分数
        for q in MODEL_CFG.quantile_levels:
            if str(q) in pred_df.columns:
                for val in pred_df[str(q)]:
                    num = int(np.clip(np.round(val), 1, 80))
                    candidate_scores[num] += WEIGHTS['prediction']['quantile']
        
        # 历史频率分数
        freq_analysis = historical_patterns.get('frequency_analysis', {})
        if 'number_freq' in freq_analysis:
            number_freq = freq_analysis['number_freq']
            max_freq = number_freq.max()
            for num in range(1, 81):
                freq_score = (number_freq.get(num, 0) / max_freq) * WEIGHTS['frequency']['historical']
                candidate_scores[num] += freq_score
        
        # 近期频率分数
        if 'recent_freq' in freq_analysis:
            recent_freq = freq_analysis['recent_freq']
            max_recent = recent_freq.max()
            for num in range(1, 81):
                recent_score = (recent_freq.get(num, 0) / max_recent) * WEIGHTS['frequency']['recent']
                candidate_scores[num] += recent_score
        
        # 趋势分数
        trend_analysis = historical_patterns.get('trend_analysis', {})
        if 'window_50' in trend_analysis:
            trend_scores = trend_analysis['window_50']
            for num, trend_score in trend_scores.items():
                candidate_scores[num] += trend_score * WEIGHTS['frequency']['trend']
        
        # 选择候选池
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        pool_size = min(60, max(30, len([s for s in candidate_scores.values() if s > 0])))
        
        return [num for num, _ in sorted_candidates[:pool_size]]
        
    def _intelligent_number_selection(self, candidates: List[int], 
                                    pred_df: pd.DataFrame,
                                    historical_patterns: Dict[str, Any],
                                    original_df: pd.DataFrame) -> List[int]:
        """智能数字选择"""
        # 计算预测分数
        prediction_scores = {num: 0.0 for num in candidates}
        
        # 基于预测值的分数
        if 'mean' in pred_df.columns:
            for val in pred_df['mean']:
                num = int(np.clip(np.round(val), 1, 80))
                if num in prediction_scores:
                    prediction_scores[num] += WEIGHTS['prediction']['mean']
        
        # 不确定性惩罚
        if len(MODEL_CFG.quantile_levels) >= 2:
            q_low = str(MODEL_CFG.quantile_levels[0])
            q_high = str(MODEL_CFG.quantile_levels[-1])
            
            if q_low in pred_df.columns and q_high in pred_df.columns:
                for i, row in pred_df.iterrows():
                    pred_num = int(np.clip(np.round(row['mean']), 1, 80))
                    if pred_num in prediction_scores:
                        uncertainty = row[q_high] - row[q_low]
                        max_uncertainty = 80.0  # 理论最大不确定性
                        penalty = (uncertainty / max_uncertainty) * WEIGHTS['prediction']['uncertainty_penalty']
                        prediction_scores[pred_num] -= penalty
        
        # 使用遗传算法优化
        if OPT_CFG.use_genetic_algorithm:
            optimizer = MultiObjectiveGeneticOptimizer(OPT_CFG)
            final_numbers = optimizer.optimize_number_selection(
                candidates, prediction_scores, historical_patterns
            )
        else:
            # 贪婪选择
            final_numbers = self._greedy_selection(candidates, prediction_scores, historical_patterns)
        
        return final_numbers
        
    def _greedy_selection(self, candidates: List[int], 
                         prediction_scores: Dict[int, float],
                         historical_patterns: Dict[str, Any]) -> List[int]:
        """贪婪选择策略"""
        # 根据分数排序
        sorted_candidates = sorted(candidates, key=lambda x: prediction_scores.get(x, 0), reverse=True)
        
        # 分布约束
        dist_stats = historical_patterns.get('distribution_analysis', {}).get('combined', {})
        target_low = max(0, min(20, int(round(dist_stats.get('low_range', 5)))))
        target_mid = max(0, min(20, int(round(dist_stats.get('mid_range', 10)))))
        target_high = max(0, min(20, int(round(dist_stats.get('high_range', 5)))))
        
        # 调整使总和为20
        total = target_low + target_mid + target_high
        if total != 20:
            diff = 20 - total
            target_mid += diff  # 优先调整中段
            target_mid = max(0, target_mid)
        
        # 按段选择
        selected = []
        low_candidates = [n for n in sorted_candidates if 1 <= n <= 20]
        mid_candidates = [n for n in sorted_candidates if 21 <= n <= 60]
        high_candidates = [n for n in sorted_candidates if 61 <= n <= 80]
        
        selected.extend(low_candidates[:target_low])
        selected.extend(mid_candidates[:target_mid])
        selected.extend(high_candidates[:target_high])
        
        # 不足则补充
        if len(selected) < 20:
            remaining = [n for n in sorted_candidates if n not in selected]
            needed = 20 - len(selected)
            selected.extend(remaining[:needed])
        
        return selected[:20]
        
    def _assess_prediction_confidence(self, final_numbers: List[int],
                                    uncertainty_metrics: Dict[str, float],
                                    historical_patterns: Dict[str, Any]) -> float:
        """评估预测置信度"""
        confidence_factors = []
        
        # 基于不确定性的置信度
        uncertainty_score = 0.0
        for metric, value in uncertainty_metrics.items():
            weight = self.uncertainty_weights.get(metric, 1.0)
            uncertainty_score += weight * value
        
        uncertainty_confidence = 1.0 - min(1.0, uncertainty_score)
        confidence_factors.append(uncertainty_confidence)
        
        # 历史一致性置信度
        freq_analysis = historical_patterns.get('frequency_analysis', {})
        if 'number_freq' in freq_analysis:
            number_freq = freq_analysis['number_freq']
            selected_freqs = [number_freq.get(num, 0) for num in final_numbers]
            avg_freq = np.mean(selected_freqs)
            max_freq = number_freq.max()
            freq_confidence = min(1.0, avg_freq / max_freq) if max_freq > 0 else 0.5
            confidence_factors.append(freq_confidence)
        
        # 分布一致性置信度
        dist_analysis = historical_patterns.get('distribution_analysis', {})
        if 'combined' in dist_analysis:
            target_dist = dist_analysis['combined']
            actual_low = sum(1 for n in final_numbers if 1 <= n <= 20)
            actual_mid = sum(1 for n in final_numbers if 21 <= n <= 60)
            actual_high = sum(1 for n in final_numbers if 61 <= n <= 80)
            actual_odd = sum(1 for n in final_numbers if n % 2 == 1)
            
            dist_errors = [
                abs(actual_low - target_dist.get('low_range', 5)) / 20,
                abs(actual_mid - target_dist.get('mid_range', 10)) / 20,
                abs(actual_high - target_dist.get('high_range', 5)) / 20,
                abs(actual_odd - target_dist.get('odd_count', 10)) / 20
            ]
            dist_confidence = 1.0 - np.mean(dist_errors)
            confidence_factors.append(dist_confidence)
        
        # 综合置信度
        final_confidence = np.mean(confidence_factors)
        return max(0.0, min(1.0, final_confidence))

# ===== 结果导出和分析 =========================================================

class ResultExporter:
    """结果导出器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def export_comprehensive_results(self, 
                                   prediction_result: Dict[str, Any],
                                   historical_patterns: Dict[str, Any],
                                   model_predictions: Dict[str, pd.DataFrame],
                                   next_date: pd.Timestamp):
        """导出综合结果"""
        logger.info("开始导出综合结果...")
        
        date_str = next_date.strftime('%Y%m%d')
        
        # 导出最终预测
        self._export_final_prediction(prediction_result, date_str)
        
        # 导出详细分析报告
        self._export_analysis_report(prediction_result, historical_patterns, date_str)
        
        # 导出模型预测详情
        self._export_model_predictions(model_predictions, date_str)
        
        # 导出可视化图表
        self._export_visualization(prediction_result, historical_patterns, date_str)
        
        logger.info("结果导出完成")
        
    def _export_final_prediction(self, prediction_result: Dict[str, Any], date_str: str):
        """导出最终预测结果"""
        final_numbers = prediction_result['final_numbers']
        confidence = prediction_result['confidence_score']
        
        # CSV格式
        csv_file = os.path.join(self.results_dir, f'final_prediction_{date_str}.csv')
        result_df = pd.DataFrame({
            'date': [date_str],
            'confidence_score': [confidence],
            **{f'number_{i+1}': [num] for i, num in enumerate(final_numbers)}
        })
        result_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # JSON格式（包含更多信息）
        json_file = os.path.join(self.results_dir, f'prediction_details_{date_str}.json')
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json_dump_safe({
                    'date': date_str,
                    'final_numbers': final_numbers,
                    'confidence_score': confidence,
                    'uncertainty_metrics': prediction_result['uncertainty_metrics'],
                    'candidate_pool_size': len(prediction_result['candidate_pool']),
                    'prediction_source': prediction_result['prediction_source']
                }, f, ensure_ascii=False, indent=2)

    def _export_analysis_report(self, prediction_result: Dict[str, Any],
                              historical_patterns: Dict[str, Any], date_str: str):
        """导出详细分析报告"""
        report_file = os.path.join(self.results_dir, f'analysis_report_{date_str}.md')
        
        final_numbers = prediction_result['final_numbers']
        confidence = prediction_result['confidence_score']
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# KL-8 智能预测分析报告\n")
            f.write(f"**预测日期**: {date_str}\n")
            f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 核心预测结果
            f.write(f"## 🎯 核心预测结果\n\n")
            f.write(f"**最终预测数字**: {final_numbers}\n")
            f.write(f"**预测置信度**: {confidence:.3f} ({'高' if confidence > 0.7 else '中' if confidence > 0.5 else '低'})\n")
            f.write(f"**预测来源**: {prediction_result['prediction_source']}\n\n")
            
            # 数字分布分析
            f.write(f"## 📊 数字分布分析\n\n")
            low_count = sum(1 for n in final_numbers if 1 <= n <= 20)
            mid_count = sum(1 for n in final_numbers if 21 <= n <= 60)
            high_count = sum(1 for n in final_numbers if 61 <= n <= 80)
            odd_count = sum(1 for n in final_numbers if n % 2 == 1)
            even_count = 20 - odd_count
            
            f.write(f"- **低段数字 (1-20)**: {low_count} 个\n")
            f.write(f"- **中段数字 (21-60)**: {mid_count} 个\n")
            f.write(f"- **高段数字 (61-80)**: {high_count} 个\n")
            f.write(f"- **奇数**: {odd_count} 个, **偶数**: {even_count} 个\n\n")
            
            # 不确定性分析
            f.write(f"## ⚠️ 不确定性分析\n\n")
            uncertainty = prediction_result['uncertainty_metrics']
            f.write(f"- **模型分歧度**: {uncertainty.get('model_disagreement', 0):.3f}\n")
            f.write(f"- **预测标准差**: {uncertainty.get('prediction_std', 0):.3f}\n")
            f.write(f"- **历史波动性**: {uncertainty.get('historical_volatility', 0):.3f}\n\n")
            
            # 历史模式洞察
            f.write(f"## 🔍 历史模式洞察\n\n")
            freq_analysis = historical_patterns.get('frequency_analysis', {})
            if 'hot_numbers' in freq_analysis:
                hot_nums = freq_analysis['hot_numbers'][:10]
                cold_nums = freq_analysis['cold_numbers'][:10]
                f.write(f"- **当前热门数字**: {hot_nums}\n")
                f.write(f"- **当前冷门数字**: {cold_nums}\n")
                
                selected_hot = [n for n in final_numbers if n in hot_nums]
                selected_cold = [n for n in final_numbers if n in cold_nums]
                f.write(f"- **预测中的热门数字**: {selected_hot} ({len(selected_hot)}个)\n")
                f.write(f"- **预测中的冷门数字**: {selected_cold} ({len(selected_cold)}个)\n\n")
            
            # 风险提示
            f.write(f"## ⚡ 风险提示\n\n")
            if confidence < 0.5:
                f.write(f"- ⚠️ **低置信度预测**: 当前预测置信度较低，建议谨慎参考\n")
            if uncertainty.get('model_disagreement', 0) > 0.3:
                f.write(f"- ⚠️ **模型分歧**: 不同模型预测存在较大分歧，增加不确定性\n")
            if len(prediction_result['candidate_pool']) < 40:
                f.write(f"- ⚠️ **候选池较小**: 可选数字范围较窄，可能影响预测质量\n")
                
            f.write(f"\n---\n*此预测仅供参考，不构成任何投注建议。请理性对待彩票游戏。*\n")
            
    def _export_model_predictions(self, model_predictions: Dict[str, pd.DataFrame], date_str: str):
        """导出模型预测详情"""
        models_file = os.path.join(self.results_dir, f'model_predictions_{date_str}.csv')
        
        all_predictions = []
        for model_name, pred_df in model_predictions.items():
            model_data = pred_df.copy()
            model_data['model'] = model_name
            model_data['date'] = date_str
            all_predictions.append(model_data)
        
        if all_predictions:
            combined_df = pd.concat(all_predictions, ignore_index=True)
            combined_df.to_csv(models_file, index=False, encoding='utf-8')
            
    def _export_visualization(self, prediction_result: Dict[str, Any],
                            historical_patterns: Dict[str, Any], date_str: str):
        """导出可视化图表（文本版）"""
        viz_file = os.path.join(self.results_dir, f'visualization_{date_str}.txt')
        
        with open(viz_file, 'w', encoding='utf-8') as f:
            f.write("KL-8 预测可视化分析\n")
            f.write("=" * 50 + "\n\n")
            
            # 数字分布直方图（文本版）
            final_numbers = prediction_result['final_numbers']
            f.write("数字分布图:\n")
            f.write("-" * 30 + "\n")
            
            ranges = [(1, 20, "低段"), (21, 60, "中段"), (61, 80, "高段")]
            for start, end, label in ranges:
                count = sum(1 for n in final_numbers if start <= n <= end)
                bar = "█" * count + "░" * (10 - count)
                f.write(f"{label:4s} |{bar}| {count:2d}\n")
            
            f.write("\n")
            
            # 置信度表盘（文本版）
            confidence = prediction_result['confidence_score']
            f.write("置信度评估:\n")
            f.write("-" * 30 + "\n")
            conf_level = int(confidence * 10)
            conf_bar = "█" * conf_level + "░" * (10 - conf_level)
            f.write(f"置信度 |{conf_bar}| {confidence:.3f}\n")

# ===== 主控制器 ==============================================================

class KL8PredictionSystem:
    """KL-8预测系统主控制器"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.predictor = AdvancedPredictor(MODEL_CFG)
        self.pattern_analyzer = PatternAnalyzer(MODEL_CFG.hist_window)
        self.prediction_generator = IntelligentPredictionGenerator()
        self.result_exporter = None
        
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """运行完整预测管道"""
        logger.info("🚀 开始KL-8深度优化预测系统...")
        
        try:
            # 步骤1: 数据加载和预处理
            logger.info("📊 步骤1: 数据加载和预处理")
            raw_df = self.data_processor.load_and_validate_data(DATA_PATH)
            processed_df = self.data_processor.preprocess_comprehensive(raw_df)
            
            # 步骤2: 历史模式分析
            logger.info("🔍 步骤2: 历史模式分析")
            historical_patterns = self.pattern_analyzer.analyze_comprehensive_patterns(raw_df)
            
            # 步骤3: 模型训练
            logger.info("🤖 步骤3: 高级模型训练")
            self.predictor.train(processed_df)
            
            # 步骤4: 预测生成
            logger.info("🎯 步骤4: 智能预测生成")
            model_predictions = self.predictor.predict_with_uncertainty(processed_df)
            
            # 步骤5: 最终预测优化
            logger.info("✨ 步骤5: 最终预测优化")
            prediction_result = self.prediction_generator.generate_final_prediction(
                model_predictions, historical_patterns, raw_df
            )
            
            # 步骤6: 结果导出
            logger.info("📤 步骤6: 结果导出和分析")
            next_date = processed_df['timestamp'].max() + pd.Timedelta(days=1)
            results_dir = os.path.join(RESULTS_ROOT, next_date.strftime('%Y%m%d'))
            
            self.result_exporter = ResultExporter(results_dir)
            self.result_exporter.export_comprehensive_results(
                prediction_result, historical_patterns, model_predictions, next_date
            )
            
            # 步骤7: 性能评估
            logger.info("📈 步骤7: 模型性能评估")
            self._run_performance_evaluation(processed_df, results_dir)
            
            # 汇总结果
            summary = {
                'status': 'success',
                'prediction_date': next_date.strftime('%Y-%m-%d'),
                'final_numbers': prediction_result['final_numbers'],
                'confidence_score': prediction_result['confidence_score'],
                'results_directory': results_dir,
                'model_count': len(model_predictions),
                'feature_count': processed_df.shape[1] - 3,  # 减去timestamp, item_id, target
                'training_samples': len(processed_df)
            }
            
            logger.info("🎉 KL-8预测系统运行成功!")
            return summary
            
        except FileNotFoundError:
            logger.critical(f"数据文件未找到: {DATA_PATH}. 请检查路径配置。")
        except Exception as e:
            logger.critical(f"预测管道执行失败: {e}", exc_info=True)
            return {
                'status': 'failure',
                'error_message': str(e)
            }

    def _run_performance_evaluation(self, data_df: pd.DataFrame, results_dir: str):
        """
        运行模型性能评估。
        使用`fit`期间生成的leaderboard，因为它反映了在多个验证窗口上的回测性能。
        """
        logger.info("开始模型性能评估...")
        try:
            if self.predictor and self.predictor.predictor:
                # 获取在训练期间生成的回测性能排行榜
                leaderboard = self.predictor.predictor.leaderboard(data_df, silent=True)
                
                if leaderboard is not None and not leaderboard.empty:
                    leaderboard_path = os.path.join(results_dir, 'model_performance_leaderboard.csv')
                    leaderboard.to_csv(leaderboard_path, index=False, encoding='utf-8')
                    logger.info(f"模型性能排行榜已保存至: {leaderboard_path}")
                    
                    # 记录性能摘要
                    logger.info("--- 模型性能摘要 ---")
                    logger.info(f"总计模型数: {len(leaderboard)}")
                    best_model = leaderboard.iloc[0]
                    logger.info(f"最佳模型: {best_model['model']}")
                    logger.info(f"  - 验证分数 (WQL): {best_model['score_val']:.4f}")
                    logger.info(f"  - 训练时间: {best_model['fit_time_s']:.2f} 秒")
                    logger.info("--------------------")
                    
                    # 可以在此处添加更详细的评估指标分析
                    
                else:
                    logger.warning("未能生成或获取模型性能排行榜。")
            else:
                logger.error("预测器未被正确初始化，无法进行性能评估。")
        
        except Exception as e:
            logger.error(f"性能评估过程中发生错误: {e}", exc_info=True)


# ===== 主执行入口 ============================================================

if __name__ == "__main__":
    try:
        # 确保结果和缓存目录存在
        os.makedirs(RESULTS_ROOT, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # 实例化并运行系统
        system = KL8PredictionSystem()
        final_summary = system.run_complete_pipeline()
        
        # 打印最终摘要
        print("\n" + "=" * 60)
        print("          KL-8 ADVANCED FORECASTING SYSTEM v2.0          ")
        print("                  -- EXECUTION SUMMARY --                ")
        print("=" * 60)
        
        if final_summary['status'] == 'success':
            for key, value in final_summary.items():
                # 格式化输出
                key_str = key.replace('_', ' ').title()
                print(f"{key_str:<25}: {value}")
        else:
            print(f"{'Status':<25}: FAILURE")
            print(f"{'Error':<25}: {final_summary.get('error_message', 'Unknown error')}")
            
        print("=" * 60)
        
    except FileNotFoundError:
        logger.critical(f"数据文件未找到: {DATA_PATH}. 请检查路径配置。")
    except Exception as e:
        logger.critical(f"系统因未知严重错误而终止: {e}", exc_info=True)
    finally:
        # 清理内存
        gc.collect()
        logger.info("系统运行结束。")