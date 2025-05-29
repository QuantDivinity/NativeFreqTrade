# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, RealParameter
from pandas import DataFrame
import pandas as pd # Added for feature engineering
import numpy as np  # Added for feature engineering
import logging     # Added for logging within feature engineering

# --------------------------------

# Get the logger for this module
logger = logging.getLogger(__name__)
# You can set the logging level for this specific logger if needed for debugging
# logging.getLogger(__name__).setLevel(logging.DEBUG)


class TSM_OHLCV_Strategy(IStrategy):
    """
    Freqtrade strategy for the TSM OHLCV GRU model.
    This strategy uses FreqAI to:
    1. Perform feature engineering directly within this class.
    2. Utilize GRUFreqaiModel for predictions.
    3. Generate buy/sell signals based on the model's DI_value (predicted return).
    """

    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "60": 0.01
    }

    stoploss = -0.10

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    timeframe = '15m' # Must match the data used for training the GRU model

    # --- FreqAI Configuration ---
    freqai_info = {
        # "feature_engineering_space": # Removed, as features are in this class
        "model_training_space": "user_data.freqaimodels.gru_freqai_model.GRUFreqaiModel",
        "live_retrain_hours": 6,
        "DI_buy_period_candles": 1, # Matches predict_length=1 for the target variable
    }

    buy_di_threshold = RealParameter(0.0, 0.05, default=0.005, space="buy", optimize=True)

    order_types = {
        "buy": "limit",
        "sell": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {
        "buy": "gtc",
        "sell": "gtc",
    }

    plot_config = {
        'main_plot': {},
        'subplots': {
            "DI": {
                 'DI_values': {'color': 'orange', 'type': 'line'}
            },
            # Example for plotting one of our new features (ensure it exists in final df)
            # "Features": {
            #      '%feat_volume_norm': {'color': 'blue', 'type': 'line'}
            # }
        }
    }

    # --- Feature Engineering (Moved from separate engine class) ---
    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        logger.info(f"Calculating features for {metadata['pair']} using feature_engineering_standard...")
        processed_df = dataframe.copy()

        predict_length = self.freqai_info.get('DI_buy_period_candles', 1)

        if not isinstance(processed_df.index, pd.DatetimeIndex):
            if 'date' in processed_df.columns:
                processed_df = processed_df.set_index('date', drop=True)
            else:
                raise ValueError("DataFrame must have a 'date' column or a DatetimeIndex.")

        # ==== 1. TRDP ====
        if not all(col in processed_df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("OHLCV data is missing from the input dataframe for TRDP calc.")

        processed_df['trdp'] = (
            processed_df['open'] + processed_df['high'] + processed_df['low'] + processed_df['close']
        ) / 4
        logger.debug(f"TRDP calculated for {metadata['pair']}.")

        # ==== 2. dv_ma5 (Daily Volume 5-day Moving Average) ====
        if 'volume' not in processed_df.columns:
            raise ValueError("'volume' column missing for dv_ma5 calculation.")

        daily_volume = processed_df['volume'].resample('D').sum(min_count=1)
        dv_ma5_daily = daily_volume.rolling(window=5, min_periods=1).mean().shift(1)
        dv_ma5_daily.name = 'dv_ma5'

        # Store original index name (usually 'date')
        original_index_name = processed_df.index.name
        # Reset index to merge, then set it back
        processed_df_reset = processed_df.reset_index()
        processed_df_reset['day_date_for_merge'] = processed_df_reset['date'].dt.normalize()

        dv_ma5_to_merge = dv_ma5_daily.reset_index()
        dv_ma5_to_merge.rename(columns={'date': 'day_date_for_merge'}, inplace=True)

        processed_df_merged = pd.merge(processed_df_reset, dv_ma5_to_merge[['day_date_for_merge', 'dv_ma5']], on='day_date_for_merge', how='left')

        processed_df_merged.drop(columns=['day_date_for_merge'], inplace=True)
        processed_df = processed_df_merged.set_index('date') # Set 'date' column back to index
        if original_index_name: # Restore original index name if it existed
             processed_df.index.name = original_index_name
        logger.debug(f"dv_ma5 calculated and merged for {metadata['pair']}.")

        # ==== 3. Normalized Volume Feature (becomes %feat_volume_norm) ====
        K_norm = 96.0
        dv_ma5_safe = processed_df['dv_ma5'].replace(0, np.nan)
        processed_df['%feat_volume_norm'] = K_norm * processed_df['volume'] / dv_ma5_safe
        processed_df['%feat_volume_norm'].fillna(0.0, inplace=True)
        logger.debug(f"Normalized Volume Feature (%feat_volume_norm) calculated for {metadata['pair']}.")

        # ==== 4. Sigma (Volatility) ====
        trdp_shifted = processed_df['trdp'].shift(1)
        trdp_shifted_safe = trdp_shifted.replace(0, np.nan)
        rtn_for_sigma = (processed_df['trdp'] / trdp_shifted_safe) - 1

        sigma_series = rtn_for_sigma.rolling(window=40, min_periods=1).std()
        sigma_series.fillna(method='ffill', inplace=True)
        sigma_series.fillna(method='bfill', inplace=True)
        sigma_epsilon = 1e-8
        sigma_series.replace(0, sigma_epsilon, inplace=True)
        sigma_series.fillna(sigma_epsilon, inplace=True)
        processed_df['sigma'] = sigma_series
        logger.debug(f"Sigma (volatility) calculated for {metadata['pair']}.")

        # ==== 5. Target Variable (becomes &DI_target) ====
        trdp_future = processed_df['trdp'].shift(-predict_length)
        trdp_safe = processed_df['trdp'].replace(0, np.nan)
        rtn_target = (trdp_future / trdp_safe) - 1

        processed_df['&DI_target'] = rtn_target / processed_df['sigma']
        logger.debug(f"Target Variable (&DI_target) calculated for {metadata['pair']}.")

        # ==== 6. Assign Final Model Features (with % prefix) ====
        processed_df['%feat_open'] = processed_df['open']
        processed_df['%feat_high'] = processed_df['high']
        processed_df['%feat_low'] = processed_df['low']
        processed_df['%feat_close'] = processed_df['close']
        # '%feat_volume_norm' is already correctly named.

        # Log NaN counts for important generated columns for debugging
        # logger.debug(f"NaN count in %feat_volume_norm for {metadata['pair']}: {processed_df['%feat_volume_norm'].isna().sum()}")
        # logger.debug(f"NaN count in sigma for {metadata['pair']}: {processed_df['sigma'].isna().sum()}")
        # logger.debug(f"NaN count in &DI_target for {metadata['pair']}: {processed_df['&DI_target'].isna().sum()} (first {predict_length} are expected NaNs)")

        return processed_df

    # NOTE: feature_engineering_expand_basic and feature_engineering_expand_all
    # are not strictly needed if feature_engineering_standard does everything.
    # FreqAI will call them if they exist. We can leave them as pass-through.
    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # logger.debug(f"Called feature_engineering_expand_basic for {metadata['pair']}. No ops.")
        return dataframe

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict) -> DataFrame:
        # logger.debug(f"Called feature_engineering_expand_all for {metadata['pair']} with period {period}. No ops.")
        return dataframe

    # --- Standard Freqtrade methods ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        All indicators are handled by FreqAI via feature_engineering_* methods.
        This method is called by Freqtrade but after FreqAI has already processed data.
        If you need to add indicators NOT for FreqAI but for regular strategy logic/plotting,
        you could add them here, but ensure they don't conflict with FreqAI's features.
        """
        # logger.debug(f"populate_indicators called for {metadata['pair']}. DI_values should be present if model predicted.")
        # if 'DI_values' in dataframe.columns:
        #     logger.info(f"DI_values in populate_indicators for {metadata['pair']}: {dataframe['DI_values'].tail(3).to_list()}")
        # else:
        #     logger.info(f"DI_values not yet in dataframe in populate_indicators for {metadata['pair']}")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define buy signals based on DI_values from FreqAI.
        """
        if 'DI_values' in dataframe.columns:
            dataframe.loc[
                (dataframe['DI_values'] > self.buy_di_threshold.value),
                'buy'] = 1
        else:
            # DI_values not available (e.g., during FreqAI startup, or if model failed)
            dataframe['buy'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define sell signals. For now, relies on ROI/stoploss.
        """
        if 'sell' not in dataframe.columns:
            dataframe['sell'] = 0 # Initialize 'sell' column if not present
        return dataframe