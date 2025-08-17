import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

class TechnicalIndicators:
    def __init__(self):
        """
        Initialize Technical Indicators calculator
        """
        pass
    
    @staticmethod
    def sma(data, window):
        """
        Simple Moving Average
        """
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        """
        Exponential Moving Average
        """
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        """
        Relative Strength Index
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """
        MACD (Moving Average Convergence Divergence)
        """
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        """
        Bollinger Bands
        """
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return pd.DataFrame({
            'Upper': upper_band,
            'Middle': sma,
            'Lower': lower_band,
            'BandWidth': (upper_band - lower_band) / sma,
            'BandPosition': (data - lower_band) / (upper_band - lower_band)
        })
    
    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        """
        Stochastic Oscillator
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'K': k_percent,
            'D': d_percent
        })
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        """
        Williams %R
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def atr(high, low, close, period=14):
        """
        Average True Range
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def adx(high, low, close, period=14):
        """
        Average Directional Index
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high - high.shift()
        dm_minus = low.shift() - low
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smooth the values
        atr_smooth = true_range.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / atr_smooth)
        di_minus = 100 * (dm_minus_smooth / atr_smooth)
        
        # Calculate DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            'DI_Plus': di_plus,
            'DI_Minus': di_minus
        })
    
    @staticmethod
    def obv(close, volume):
        """
        On-Balance Volume
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def cci(high, low, close, period=20):
        """
        Commodity Channel Index
        """
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
        
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def momentum(data, period=10):
        """
        Momentum indicator
        """
        return data / data.shift(period) - 1
    
    @staticmethod
    def roc(data, period=10):
        """
        Rate of Change
        """
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    @staticmethod
    def trix(data, period=14):
        """
        TRIX indicator
        """
        ema1 = data.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        
        trix = (ema3 / ema3.shift(1) - 1) * 10000
        return trix
    
    def calculate_all_indicators(self, df):
        """
        Calculate all technical indicators for a given dataframe
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with all technical indicators
        """
        result = df.copy()
        
        # Price-based indicators
        result['SMA_10'] = self.sma(df['Close'], 10)
        result['SMA_20'] = self.sma(df['Close'], 20)
        result['SMA_50'] = self.sma(df['Close'], 50)
        result['SMA_200'] = self.sma(df['Close'], 200)
        
        result['EMA_10'] = self.ema(df['Close'], 10)
        result['EMA_20'] = self.ema(df['Close'], 20)
        result['EMA_50'] = self.ema(df['Close'], 50)
        
        # RSI
        result['RSI'] = self.rsi(df['Close'])
        
        # MACD
        macd_data = self.macd(df['Close'])
        result['MACD'] = macd_data['MACD']
        result['MACD_Signal'] = macd_data['Signal']
        result['MACD_Histogram'] = macd_data['Histogram']
        
        # Bollinger Bands
        bb_data = self.bollinger_bands(df['Close'])
        result['BB_Upper'] = bb_data['Upper']
        result['BB_Middle'] = bb_data['Middle']
        result['BB_Lower'] = bb_data['Lower']
        result['BB_Width'] = bb_data['BandWidth']
        result['BB_Position'] = bb_data['BandPosition']
        
        # Stochastic
        stoch_data = self.stochastic(df['High'], df['Low'], df['Close'])
        result['Stoch_K'] = stoch_data['K']
        result['Stoch_D'] = stoch_data['D']
        
        # Williams %R
        result['Williams_R'] = self.williams_r(df['High'], df['Low'], df['Close'])
        
        # ATR
        result['ATR'] = self.atr(df['High'], df['Low'], df['Close'])
        
        # ADX
        adx_data = self.adx(df['High'], df['Low'], df['Close'])
        result['ADX'] = adx_data['ADX']
        result['DI_Plus'] = adx_data['DI_Plus']
        result['DI_Minus'] = adx_data['DI_Minus']
        
        # Volume indicators
        if 'Volume' in df.columns:
            result['OBV'] = self.obv(df['Close'], df['Volume'])
        
        # CCI
        result['CCI'] = self.cci(df['High'], df['Low'], df['Close'])
        
        # Momentum indicators
        result['Momentum'] = self.momentum(df['Close'])
        result['ROC'] = self.roc(df['Close'])
        result['TRIX'] = self.trix(df['Close'])
        
        # Price ratios and patterns
        result['High_Low_Ratio'] = df['High'] / df['Low']
        result['Close_Open_Ratio'] = df['Close'] / df['Open']
        result['Body_Size'] = abs(df['Close'] - df['Open']) / df['Open']
        result['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Open']
        result['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Open']
        
        # Volatility measures
        result['Price_Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        if 'Volume' in df.columns:
            result['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            result['Volume_Ratio'] = df['Volume'] / result['Volume_MA']
        
        return result
    
    def detect_patterns(self, df):
        """
        Detect common chart patterns
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with pattern signals
        """
        patterns = pd.DataFrame(index=df.index)
        
        # Calculate indicators needed for patterns
        indicators = self.calculate_all_indicators(df)
        
        # Doji pattern
        body_size = abs(df['Close'] - df['Open'])
        avg_body = body_size.rolling(window=10).mean()
        patterns['Doji'] = (body_size < avg_body * 0.1).astype(int)
        
        # Hammer pattern
        body = abs(df['Close'] - df['Open'])
        lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
        upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
        
        patterns['Hammer'] = (
            (lower_shadow >= 2 * body) & 
            (upper_shadow <= 0.1 * body) &
            (df['Close'] > df['Open'])
        ).astype(int)
        
        # Golden Cross (50 SMA crosses above 200 SMA)
        patterns['Golden_Cross'] = (
            (indicators['SMA_50'] > indicators['SMA_200']) &
            (indicators['SMA_50'].shift(1) <= indicators['SMA_200'].shift(1))
        ).astype(int)
        
        # Death Cross (50 SMA crosses below 200 SMA)
        patterns['Death_Cross'] = (
            (indicators['SMA_50'] < indicators['SMA_200']) &
            (indicators['SMA_50'].shift(1) >= indicators['SMA_200'].shift(1))
        ).astype(int)
        
        # Bullish MACD crossover
        patterns['MACD_Bullish'] = (
            (indicators['MACD'] > indicators['MACD_Signal']) &
            (indicators['MACD'].shift(1) <= indicators['MACD_Signal'].shift(1))
        ).astype(int)
        
        # Bearish MACD crossover
        patterns['MACD_Bearish'] = (
            (indicators['MACD'] < indicators['MACD_Signal']) &
            (indicators['MACD'].shift(1) >= indicators['MACD_Signal'].shift(1))
        ).astype(int)
        
        # RSI oversold/overbought
        patterns['RSI_Oversold'] = (indicators['RSI'] < 30).astype(int)
        patterns['RSI_Overbought'] = (indicators['RSI'] > 70).astype(int)
        
        # Bollinger Band squeeze
        bb_width = indicators['BB_Width']
        bb_width_ma = bb_width.rolling(window=20).mean()
        patterns['BB_Squeeze'] = (bb_width < bb_width_ma * 0.5).astype(int)
        
        # Bollinger Band breakout
        patterns['BB_Upper_Break'] = (df['Close'] > indicators['BB_Upper']).astype(int)
        patterns['BB_Lower_Break'] = (df['Close'] < indicators['BB_Lower']).astype(int)
        
        # Volume spike
        if 'Volume' in df.columns:
            volume_ma = df['Volume'].rolling(window=20).mean()
            patterns['Volume_Spike'] = (df['Volume'] > volume_ma * 2).astype(int)
        
        return patterns
    
    def generate_signals(self, df):
        """
        Generate trading signals based on technical indicators
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with buy/sell signals
        """
        indicators = self.calculate_all_indicators(df)
        patterns = self.detect_patterns(df)
        
        signals = pd.DataFrame(index=df.index)
        signals['Buy_Signal'] = 0
        signals['Sell_Signal'] = 0
        
        # RSI signals
        rsi_buy = (indicators['RSI'] < 30) & (indicators['RSI'].shift(1) >= 30)
        rsi_sell = (indicators['RSI'] > 70) & (indicators['RSI'].shift(1) <= 70)
        
        # MACD signals
        macd_buy = (indicators['MACD'] > indicators['MACD_Signal']) & (indicators['MACD'].shift(1) <= indicators['MACD_Signal'].shift(1))
        macd_sell = (indicators['MACD'] < indicators['MACD_Signal']) & (indicators['MACD'].shift(1) >= indicators['MACD_Signal'].shift(1))
        
        # Moving average signals
        ma_buy = (df['Close'] > indicators['SMA_50']) & (indicators['SMA_50'] > indicators['SMA_200'])
        ma_sell = (df['Close'] < indicators['SMA_50']) & (indicators['SMA_50'] < indicators['SMA_200'])
        
        # Bollinger Band signals
        bb_buy = (df['Close'] < indicators['BB_Lower']) & (indicators['RSI'] < 30)
        bb_sell = (df['Close'] > indicators['BB_Upper']) & (indicators['RSI'] > 70)
        
        # Combine signals
        signals['Buy_Signal'] = (rsi_buy | macd_buy | bb_buy).astype(int)
        signals['Sell_Signal'] = (rsi_sell | macd_sell | bb_sell).astype(int)
        
        # Add signal strength
        buy_strength = (rsi_buy.astype(int) + macd_buy.astype(int) + 
                       ma_buy.astype(int) + bb_buy.astype(int))
        sell_strength = (rsi_sell.astype(int) + macd_sell.astype(int) + 
                        ma_sell.astype(int) + bb_sell.astype(int))
        
        signals['Buy_Strength'] = buy_strength
        signals['Sell_Strength'] = sell_strength
        
        return signals
    
    def calculate_support_resistance(self, df, window=20):
        """
        Calculate support and resistance levels
        
        Args:
            df: DataFrame with OHLCV data
            window: Rolling window for calculations
        
        Returns:
            DataFrame with support and resistance levels
        """
        levels = pd.DataFrame(index=df.index)
        
        # Rolling min/max for support/resistance
        levels['Support'] = df['Low'].rolling(window=window).min()
        levels['Resistance'] = df['High'].rolling(window=window).max()
        
        # Pivot points
        levels['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        levels['R1'] = 2 * levels['Pivot'] - df['Low']
        levels['R2'] = levels['Pivot'] + (df['High'] - df['Low'])
        levels['S1'] = 2 * levels['Pivot'] - df['High']
        levels['S2'] = levels['Pivot'] - (df['High'] - df['Low'])
        
        return levels
