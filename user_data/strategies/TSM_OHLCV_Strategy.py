# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, RealParameter
from pandas import DataFrame
# --------------------------------

class TSM_OHLCV_Strategy(IStrategy):
    """
    Freqtrade strategy for the TSM OHLCV GRU model.
    This strategy uses FreqAI to:
    1. Employ TSM_OHLCV_FeatureEngine for feature engineering.
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

    timeframe = '15m'

    freqai_info = {
        "feature_engineering_space": "user_data.freqaimodels.tsm_ohlcv_feature_engine.TSM_OHLCV_FeatureEngine",
        "model_training_space": "user_data.freqaimodels.gru_freqai_model.GRUFreqaiModel",
        "live_retrain_hours": 6,
        "DI_buy_period_candles": 1,
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
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        All indicators are handled by FreqAI.
        """
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define buy signals based on DI_values from FreqAI.
        """
        dataframe.loc[
            (dataframe['DI_values'] > self.buy_di_threshold.value),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define sell signals. For now, relies on ROI/stoploss.
        """
        if 'sell' not in dataframe.columns:
            dataframe['sell'] = 0
        return dataframe
