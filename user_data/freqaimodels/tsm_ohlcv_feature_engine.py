import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from freqtrade.freqai.base_models.IFreqaiFeatureEngine import IFreqaiFeatureEngine, collect_features
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class TSM_OHLCV_FeatureEngine(IFreqaiFeatureEngine):
    """
    Feature engineering for the TSM OHLCV GRU model.
    Replicates logic from the q-SQL query qry_15mkbarXy.
    Features: open, high, low, close, volume_normalized_by_dv_ma5_and_scaled
    Target: predict_rtn = rtn_shifted / sigma
    """

    @property
    def feature_names(self) -> List[str]:
        return ['feat_open', 'feat_high', 'feat_low', 'feat_close', 'feat_volume_norm']

    @property
    def label_names(self) -> List[str]:
        return ['DI_target']

    def define_data_schema(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """
        Define the schema for the input data.
        """
        data_schema: Dict[str, str] = {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        }
        return data_schema

    def define_label_schema(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """
        Define the schema for the labels.
        """
        label_schema: Dict[str, str] = {
            "DI_target": "float64"
        }
        return label_schema

    @collect_features
    def populate_features(self, datakichen: FreqaiDataKitchen, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Populate features and labels.
        """
        dataframe = datakichen.get_raw_data() # This should have a DatetimeIndex named 'date'
        processed_df = dataframe.copy()

        # Get predict_length from freqai_info, default to 1 if not found
        predict_length = self.freqai_info.get('DI_buy_period_candles', 1)
        if predict_length != 1:
            logger.warning(f"Custom predict_length={predict_length} used for target. Original model used 1.")

        if not isinstance(processed_df.index, pd.DatetimeIndex):
            if 'date' in processed_df.columns:
                processed_df = processed_df.set_index('date', drop=True)
            else:
                raise ValueError("DataFrame must have a 'date' column or a DatetimeIndex.")

        # ==== 1. TRDP ====
        if not all(col in processed_df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("OHLC data is missing from the input dataframe.")
        
        processed_df['trdp'] = (
            processed_df['open'] + processed_df['high'] + processed_df['low'] + processed_df['close']
        ) / 4
        logger.info("Calculated TRDP (Typical Price).")

        # ==== 2. dv_ma5 (Daily Volume 5-day Moving Average) ====
        daily_volume = processed_df['volume'].resample('D').sum(min_count=1)
        dv_ma5_daily = daily_volume.rolling(window=5, min_periods=1).mean().shift(1)
        dv_ma5_daily.name = 'dv_ma5'
        processed_df['day_date_for_dvma5_merge'] = processed_df.index.normalize()
        dv_ma5_to_merge = dv_ma5_daily.reset_index()
        dv_ma5_to_merge.rename(columns={'date': 'day_date_for_dvma5_merge'}, inplace=True)
        processed_df = pd.merge(processed_df, dv_ma5_to_merge, on='day_date_for_dvma5_merge', how='left')
        processed_df.drop(columns=['day_date_for_dvma5_merge'], inplace=True)
        logger.info("Calculated dv_ma5 and merged to 15m data.")

        # ==== 3. Normalized Volume Feature ====
        K_norm = 96.0
        processed_df['dv_ma5_safe'] = processed_df['dv_ma5'].replace(0, np.nan)
        processed_df['feat_volume_norm'] = K_norm * processed_df['volume'] / processed_df['dv_ma5_safe']
        processed_df['feat_volume_norm'].fillna(0.0, inplace=True)
        processed_df.drop(columns=['dv_ma5_safe'], inplace=True)
        logger.info("Calculated Normalized Volume Feature (feat_volume_norm).")

        # ==== 4. Sigma (Volatility) ====
        if 'trdp' not in processed_df.columns:
            raise ValueError("TRDP column is missing, cannot calculate sigma.")
        trdp_shifted = processed_df['trdp'].shift(1)
        trdp_shifted_safe = trdp_shifted.replace(0, np.nan) 
        rtn_for_sigma = (processed_df['trdp'] / trdp_shifted_safe) - 1
        processed_df['sigma'] = rtn_for_sigma.rolling(window=40, min_periods=1).std() 
        processed_df['sigma'].fillna(method='ffill', inplace=True)
        processed_df['sigma'].fillna(method='bfill', inplace=True)
        sigma_epsilon = 1e-8
        processed_df['sigma'].replace(0, sigma_epsilon, inplace=True)
        processed_df['sigma'].fillna(sigma_epsilon, inplace=True)
        logger.info("Calculated Sigma (volatility).")

        # ==== 5. Target Variable (DI_target) ====
        if 'trdp' not in processed_df.columns or 'sigma' not in processed_df.columns:
            raise ValueError("TRDP or sigma column is missing, cannot calculate DI_target.")
        trdp_future = processed_df['trdp'].shift(-predict_length)
        trdp_safe = processed_df['trdp'].replace(0, np.nan)
        rtn_target = (trdp_future / trdp_safe) - 1
        processed_df['DI_target'] = rtn_target / processed_df['sigma']
        logger.info("Calculated Target Variable (DI_target).")

        # ==== 6. Select Final Features & Labels ====
        processed_df['feat_open'] = processed_df['open']
        processed_df['feat_high'] = processed_df['high']
        processed_df['feat_low'] = processed_df['low']
        processed_df['feat_close'] = processed_df['close']

        missing_feature_cols = [col for col in self.feature_names if col not in processed_df.columns]
        if missing_feature_cols:
            err_msg = f"Critical error: The following feature columns were expected but not found: {missing_feature_cols}. Review calculations."
            logger.error(err_msg)
            raise ValueError(err_msg)

        for col in self.label_names:
            if col not in processed_df.columns:
                logger.warning(f"Label column {col} was not found in processed_df. Creating it and filling with np.nan.")
                processed_df[col] = np.nan
        
        features = processed_df[self.feature_names].copy()
        labels = processed_df[self.label_names].copy()
        
        logger.info(f"Final features returned: {features.columns.tolist()}")
        logger.info(f"Final labels returned: {labels.columns.tolist()}")

        return features, labels
