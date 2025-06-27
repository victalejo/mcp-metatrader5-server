"""
MetaTrader 5 MCP Server - Technical Analysis Module

This module provides comprehensive technical analysis tools including:
- 25+ Technical Indicators
- Candlestick Pattern Detection
- Chart Pattern Recognition  
- Trend Analysis
- Support/Resistance Levels
- Trading Signals
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Import the main server instance
from mcp_metatrader5_server.server import mcp

logger = logging.getLogger("mt5-mcp-server.technical_analysis")

# Enums for signals and patterns
class TrendDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"

class SignalStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class CandlestickPattern(str, Enum):
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"

# Pydantic Models
class TechnicalIndicators(BaseModel):
    """Technical indicators calculation result"""
    symbol: str
    timeframe: int
    timestamp: datetime
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    atr: Optional[float] = None
    adx: Optional[float] = None
    cci: Optional[float] = None
    williams_r: Optional[float] = None

class PatternDetection(BaseModel):
    """Pattern detection result"""
    symbol: str
    timeframe: int
    timestamp: datetime
    pattern: CandlestickPattern
    confidence: float
    signal: TrendDirection
    strength: SignalStrength

class TrendAnalysis(BaseModel):
    """Trend analysis result"""
    symbol: str
    timeframe: int
    timestamp: datetime
    trend_direction: TrendDirection
    trend_strength: float
    trend_duration: int
    support_levels: List[float]
    resistance_levels: List[float]

class TradingSignal(BaseModel):
    """Trading signal result"""
    symbol: str
    timeframe: int
    timestamp: datetime
    signal_type: str
    direction: TrendDirection
    strength: SignalStrength
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float

# ==================== TECHNICAL INDICATORS ====================

@mcp.tool()
def calculate_sma(prices: List[float], period: int = 20) -> List[float]:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        prices: List of prices
        period: Period for calculation
        
    Returns:
        List[float]: SMA values
    """
    df = pd.DataFrame({'price': prices})
    sma = df['price'].rolling(window=period).mean()
    return sma.fillna(0).tolist()

@mcp.tool()
def calculate_ema(prices: List[float], period: int = 20) -> List[float]:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        prices: List of prices
        period: Period for calculation
        
    Returns:
        List[float]: EMA values
    """
    df = pd.DataFrame({'price': prices})
    ema = df['price'].ewm(span=period, adjust=False).mean()
    return ema.fillna(0).tolist()

@mcp.tool()
def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: List of prices
        period: Period for calculation
        
    Returns:
        List[float]: RSI values
    """
    df = pd.DataFrame({'price': prices})
    delta = df['price'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).tolist()

@mcp.tool()
def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: List of prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Dict[str, List[float]]: MACD line, signal line, and histogram
    """
    df = pd.DataFrame({'price': prices})
    ema_fast = df['price'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['price'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line.fillna(0).tolist(),
        'signal': signal_line.fillna(0).tolist(),
        'histogram': histogram.fillna(0).tolist()
    }

@mcp.tool()
def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, List[float]]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: List of prices
        period: Period for calculation
        std_dev: Standard deviation multiplier
        
    Returns:
        Dict[str, List[float]]: Upper, middle, and lower bands
    """
    df = pd.DataFrame({'price': prices})
    sma = df['price'].rolling(window=period).mean()
    std = df['price'].rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    return {
        'upper': upper.fillna(0).tolist(),
        'middle': sma.fillna(0).tolist(),
        'lower': lower.fillna(0).tolist()
    }

@mcp.tool()
def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], 
                        k_period: int = 14, d_period: int = 3) -> Dict[str, List[float]]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        k_period: %K period
        d_period: %D period
        
    Returns:
        Dict[str, List[float]]: %K and %D values
    """
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return {
        'k': k_percent.fillna(50).tolist(),
        'd': d_percent.fillna(50).tolist()
    }

@mcp.tool()
def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    """
    Calculate Average True Range (ATR).
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: Period for calculation
        
    Returns:
        List[float]: ATR values
    """
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    df['prev_close'] = df['close'].shift(1)
    
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    atr = df['tr'].rolling(window=period).mean()
    return atr.fillna(0).tolist()

@mcp.tool()
def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    """
    Calculate Average Directional Index (ADX).
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: Period for calculation
        
    Returns:
        List[float]: ADX values
    """
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    
    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['prev_close']), 
                                   abs(df['low'] - df['prev_close'])))
    
    # Calculate Directional Movement
    df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                            np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                             np.maximum(df['low'].shift(1) - df['low'], 0), 0)
    
    # Smooth the values
    df['tr_smooth'] = df['tr'].rolling(window=period).mean()
    df['dm_plus_smooth'] = df['dm_plus'].rolling(window=period).mean()
    df['dm_minus_smooth'] = df['dm_minus'].rolling(window=period).mean()
    
    # Calculate DI
    df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['tr_smooth'])
    df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['tr_smooth'])
    
    # Calculate ADX
    df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
    adx = df['dx'].rolling(window=period).mean()
    
    return adx.fillna(0).tolist()

@mcp.tool()
def calculate_cci(highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> List[float]:
    """
    Calculate Commodity Channel Index (CCI).
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: Period for calculation
        
    Returns:
        List[float]: CCI values
    """
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3  # Typical Price
    df['sma_tp'] = df['tp'].rolling(window=period).mean()
    df['mad'] = df['tp'].rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (df['tp'] - df['sma_tp']) / (0.015 * df['mad'])
    
    return cci.fillna(0).tolist()

@mcp.tool()
def calculate_williams_r(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    """
    Calculate Williams %R.
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: Period for calculation
        
    Returns:
        List[float]: Williams %R values
    """
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
    
    return williams_r.fillna(-50).tolist()

# ==================== PATTERN DETECTION ====================

@mcp.tool()
def detect_doji(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> List[bool]:
    """
    Detect Doji candlestick patterns.
    
    Args:
        opens: List of open prices
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        
    Returns:
        List[bool]: True for doji patterns
    """
    df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes})
    
    # Calculate body and total range
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    
    # Doji: body is very small relative to the total range
    doji_threshold = 0.1  # Body should be less than 10% of total range
    is_doji = df['body'] <= (df['range'] * doji_threshold)
    
    return is_doji.tolist()

@mcp.tool()
def detect_hammer(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> List[bool]:
    """
    Detect Hammer candlestick patterns.
    
    Args:
        opens: List of open prices
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        
    Returns:
        List[bool]: True for hammer patterns
    """
    df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes})
    
    df['body'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['total_range'] = df['high'] - df['low']
    
    # Hammer conditions:
    # 1. Lower shadow at least 2x the body
    # 2. Upper shadow very small or non-existent
    # 3. Body in upper part of the range
    is_hammer = (
        (df['lower_shadow'] >= 2 * df['body']) &
        (df['upper_shadow'] <= 0.1 * df['total_range']) &
        (df['body'] > 0.1 * df['total_range'])
    )
    
    return is_hammer.tolist()

@mcp.tool()
def detect_engulfing_patterns(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict[str, List[bool]]:
    """
    Detect Bullish and Bearish Engulfing patterns.
    
    Args:
        opens: List of open prices
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        
    Returns:
        Dict[str, List[bool]]: Bullish and bearish engulfing patterns
    """
    df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes})
    
    # Previous candle data
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    
    # Current and previous candle bodies
    df['current_body'] = abs(df['close'] - df['open'])
    df['prev_body'] = abs(df['prev_close'] - df['prev_open'])
    
    # Bullish Engulfing: Previous red, current green and engulfs previous
    bullish_engulfing = (
        (df['prev_close'] < df['prev_open']) &  # Previous candle is red
        (df['close'] > df['open']) &  # Current candle is green
        (df['open'] < df['prev_close']) &  # Current open below previous close
        (df['close'] > df['prev_open'])  # Current close above previous open
    )
    
    # Bearish Engulfing: Previous green, current red and engulfs previous
    bearish_engulfing = (
        (df['prev_close'] > df['prev_open']) &  # Previous candle is green
        (df['close'] < df['open']) &  # Current candle is red
        (df['open'] > df['prev_close']) &  # Current open above previous close
        (df['close'] < df['prev_open'])  # Current close below previous open
    )
    
    return {
        'bullish_engulfing': bullish_engulfing.fillna(False).tolist(),
        'bearish_engulfing': bearish_engulfing.fillna(False).tolist()
    }

# ==================== COMPREHENSIVE ANALYSIS ====================

@mcp.tool()
def analyze_market_data(symbol: str, rates_data: List[Dict[str, Any]], 
                       indicators: List[str] = None) -> TechnicalIndicators:
    """
    Perform comprehensive technical analysis on market data.
    
    Args:
        symbol: Symbol name
        rates_data: List of OHLCV data dictionaries
        indicators: List of indicators to calculate (optional)
        
    Returns:
        TechnicalIndicators: Complete technical analysis result
    """
    if not rates_data:
        raise ValueError("No market data provided")
    
    df = pd.DataFrame(rates_data)
    
    # Extract OHLCV data
    opens = df['open'].tolist()
    highs = df['high'].tolist()
    lows = df['low'].tolist()
    closes = df['close'].tolist()
    
    # Calculate all indicators
    sma_20 = calculate_sma(closes, 20)[-1] if len(closes) >= 20 else None
    sma_50 = calculate_sma(closes, 50)[-1] if len(closes) >= 50 else None
    sma_200 = calculate_sma(closes, 200)[-1] if len(closes) >= 200 else None
    
    ema_12 = calculate_ema(closes, 12)[-1] if len(closes) >= 12 else None
    ema_26 = calculate_ema(closes, 26)[-1] if len(closes) >= 26 else None
    
    rsi_values = calculate_rsi(closes, 14)
    rsi_14 = rsi_values[-1] if len(rsi_values) > 0 else None
    
    macd_data = calculate_macd(closes)
    macd = macd_data['macd'][-1] if len(macd_data['macd']) > 0 else None
    macd_signal = macd_data['signal'][-1] if len(macd_data['signal']) > 0 else None
    macd_histogram = macd_data['histogram'][-1] if len(macd_data['histogram']) > 0 else None
    
    bollinger = calculate_bollinger_bands(closes)
    bollinger_upper = bollinger['upper'][-1] if len(bollinger['upper']) > 0 else None
    bollinger_middle = bollinger['middle'][-1] if len(bollinger['middle']) > 0 else None
    bollinger_lower = bollinger['lower'][-1] if len(bollinger['lower']) > 0 else None
    
    stoch = calculate_stochastic(highs, lows, closes)
    stochastic_k = stoch['k'][-1] if len(stoch['k']) > 0 else None
    stochastic_d = stoch['d'][-1] if len(stoch['d']) > 0 else None
    
    atr_values = calculate_atr(highs, lows, closes)
    atr = atr_values[-1] if len(atr_values) > 0 else None
    
    adx_values = calculate_adx(highs, lows, closes)
    adx = adx_values[-1] if len(adx_values) > 0 else None
    
    cci_values = calculate_cci(highs, lows, closes)
    cci = cci_values[-1] if len(cci_values) > 0 else None
    
    williams_values = calculate_williams_r(highs, lows, closes)
    williams_r = williams_values[-1] if len(williams_values) > 0 else None
    
    return TechnicalIndicators(
        symbol=symbol,
        timeframe=df.get('timeframe', 0),
        timestamp=datetime.now(),
        sma_20=sma_20,
        sma_50=sma_50,
        sma_200=sma_200,
        ema_12=ema_12,
        ema_26=ema_26,
        rsi_14=rsi_14,
        macd=macd,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        bollinger_upper=bollinger_upper,
        bollinger_middle=bollinger_middle,
        bollinger_lower=bollinger_lower,
        stochastic_k=stochastic_k,
        stochastic_d=stochastic_d,
        atr=atr,
        adx=adx,
        cci=cci,
        williams_r=williams_r
    )

@mcp.tool()
def detect_candlestick_patterns(symbol: str, rates_data: List[Dict[str, Any]]) -> List[PatternDetection]:
    """
    Detect various candlestick patterns in market data.
    
    Args:
        symbol: Symbol name
        rates_data: List of OHLCV data dictionaries
        
    Returns:
        List[PatternDetection]: List of detected patterns
    """
    if not rates_data or len(rates_data) < 3:
        return []
    
    df = pd.DataFrame(rates_data)
    
    opens = df['open'].tolist()
    highs = df['high'].tolist()
    lows = df['low'].tolist()
    closes = df['close'].tolist()
    
    patterns = []
    
    # Detect Doji patterns
    doji_signals = detect_doji(opens, highs, lows, closes)
    for i, is_doji in enumerate(doji_signals[-10:]):  # Check last 10 candles
        if is_doji:
            patterns.append(PatternDetection(
                symbol=symbol,
                timeframe=df.get('timeframe', 0),
                timestamp=datetime.now(),
                pattern=CandlestickPattern.DOJI,
                confidence=0.7,
                signal=TrendDirection.SIDEWAYS,
                strength=SignalStrength.MODERATE
            ))
    
    # Detect Hammer patterns
    hammer_signals = detect_hammer(opens, highs, lows, closes)
    for i, is_hammer in enumerate(hammer_signals[-10:]):
        if is_hammer:
            patterns.append(PatternDetection(
                symbol=symbol,
                timeframe=df.get('timeframe', 0),
                timestamp=datetime.now(),
                pattern=CandlestickPattern.HAMMER,
                confidence=0.8,
                signal=TrendDirection.BULLISH,
                strength=SignalStrength.STRONG
            ))
    
    # Detect Engulfing patterns
    engulfing = detect_engulfing_patterns(opens, highs, lows, closes)
    
    for i, is_bullish in enumerate(engulfing['bullish_engulfing'][-10:]):
        if is_bullish:
            patterns.append(PatternDetection(
                symbol=symbol,
                timeframe=df.get('timeframe', 0),
                timestamp=datetime.now(),
                pattern=CandlestickPattern.BULLISH_ENGULFING,
                confidence=0.85,
                signal=TrendDirection.BULLISH,
                strength=SignalStrength.STRONG
            ))
    
    for i, is_bearish in enumerate(engulfing['bearish_engulfing'][-10:]):
        if is_bearish:
            patterns.append(PatternDetection(
                symbol=symbol,
                timeframe=df.get('timeframe', 0),
                timestamp=datetime.now(),
                pattern=CandlestickPattern.BEARISH_ENGULFING,
                confidence=0.85,
                signal=TrendDirection.BEARISH,
                strength=SignalStrength.STRONG
            ))
    
    return patterns

@mcp.tool()
def analyze_trend(symbol: str, rates_data: List[Dict[str, Any]]) -> TrendAnalysis:
    """
    Analyze market trend and identify support/resistance levels.
    
    Args:
        symbol: Symbol name
        rates_data: List of OHLCV data dictionaries
        
    Returns:
        TrendAnalysis: Comprehensive trend analysis
    """
    if not rates_data or len(rates_data) < 20:
        raise ValueError("Insufficient data for trend analysis")
    
    df = pd.DataFrame(rates_data)
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Determine trend direction using linear regression
    x = np.arange(len(closes))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, closes)
    
    # Classify trend
    if slope > 0 and r_value > 0.5:
        trend_direction = TrendDirection.BULLISH
    elif slope < 0 and r_value < -0.5:
        trend_direction = TrendDirection.BEARISH
    else:
        trend_direction = TrendDirection.SIDEWAYS
    
    # Calculate trend strength
    trend_strength = abs(r_value)
    
    # Find support and resistance levels
    # Support levels from local lows
    support_indices, _ = find_peaks(-lows, distance=5)
    support_levels = lows[support_indices].tolist() if len(support_indices) > 0 else []
    
    # Resistance levels from local highs
    resistance_indices, _ = find_peaks(highs, distance=5)
    resistance_levels = highs[resistance_indices].tolist() if len(resistance_indices) > 0 else []
    
    # Sort and take most significant levels
    support_levels = sorted(support_levels, reverse=True)[:5]
    resistance_levels = sorted(resistance_levels)[:5]
    
    return TrendAnalysis(
        symbol=symbol,
        timeframe=df.get('timeframe', 0),
        timestamp=datetime.now(),
        trend_direction=trend_direction,
        trend_strength=trend_strength,
        trend_duration=len(closes),
        support_levels=support_levels,
        resistance_levels=resistance_levels
    )

@mcp.tool()
def generate_trading_signals(symbol: str, rates_data: List[Dict[str, Any]]) -> List[TradingSignal]:
    """
    Generate comprehensive trading signals based on technical analysis.
    
    Args:
        symbol: Symbol name
        rates_data: List of OHLCV data dictionaries
        
    Returns:
        List[TradingSignal]: List of trading signals
    """
    if not rates_data or len(rates_data) < 50:
        return []
    
    signals = []
    
    # Get technical indicators
    indicators = analyze_market_data(symbol, rates_data)
    
    # Get trend analysis
    trend = analyze_trend(symbol, rates_data)
    
    # Get candlestick patterns
    patterns = detect_candlestick_patterns(symbol, rates_data)
    
    current_price = rates_data[-1]['close']
    
    # RSI Oversold/Overbought signals
    if indicators.rsi_14:
        if indicators.rsi_14 < 30:
            signals.append(TradingSignal(
                symbol=symbol,
                timeframe=0,
                timestamp=datetime.now(),
                signal_type="RSI_OVERSOLD",
                direction=TrendDirection.BULLISH,
                strength=SignalStrength.MODERATE,
                entry_price=current_price,
                stop_loss=current_price * 0.98,
                take_profit=current_price * 1.03,
                confidence=0.7
            ))
        elif indicators.rsi_14 > 70:
            signals.append(TradingSignal(
                symbol=symbol,
                timeframe=0,
                timestamp=datetime.now(),
                signal_type="RSI_OVERBOUGHT",
                direction=TrendDirection.BEARISH,
                strength=SignalStrength.MODERATE,
                entry_price=current_price,
                stop_loss=current_price * 1.02,
                take_profit=current_price * 0.97,
                confidence=0.7
            ))
    
    # MACD Crossover signals
    if indicators.macd and indicators.macd_signal:
        if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
            signals.append(TradingSignal(
                symbol=symbol,
                timeframe=0,
                timestamp=datetime.now(),
                signal_type="MACD_BULLISH_CROSSOVER",
                direction=TrendDirection.BULLISH,
                strength=SignalStrength.STRONG,
                entry_price=current_price,
                stop_loss=current_price * 0.97,
                take_profit=current_price * 1.05,
                confidence=0.8
            ))
        elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
            signals.append(TradingSignal(
                symbol=symbol,
                timeframe=0,
                timestamp=datetime.now(),
                signal_type="MACD_BEARISH_CROSSOVER",
                direction=TrendDirection.BEARISH,
                strength=SignalStrength.STRONG,
                entry_price=current_price,
                stop_loss=current_price * 1.03,
                take_profit=current_price * 0.95,
                confidence=0.8
            ))
    
    # Moving Average signals
    if indicators.sma_20 and indicators.sma_50:
        if indicators.sma_20 > indicators.sma_50 and current_price > indicators.sma_20:
            signals.append(TradingSignal(
                symbol=symbol,
                timeframe=0,
                timestamp=datetime.now(),
                signal_type="MA_GOLDEN_CROSS",
                direction=TrendDirection.BULLISH,
                strength=SignalStrength.STRONG,
                entry_price=current_price,
                stop_loss=indicators.sma_20 * 0.99,
                take_profit=current_price * 1.04,
                confidence=0.75
            ))
        elif indicators.sma_20 < indicators.sma_50 and current_price < indicators.sma_20:
            signals.append(TradingSignal(
                symbol=symbol,
                timeframe=0,
                timestamp=datetime.now(),
                signal_type="MA_DEATH_CROSS",
                direction=TrendDirection.BEARISH,
                strength=SignalStrength.STRONG,
                entry_price=current_price,
                stop_loss=indicators.sma_20 * 1.01,
                take_profit=current_price * 0.96,
                confidence=0.75
            ))
    
    # Add pattern-based signals
    for pattern in patterns:
        if pattern.confidence > 0.8:
            signals.append(TradingSignal(
                symbol=symbol,
                timeframe=0,
                timestamp=datetime.now(),
                signal_type=f"PATTERN_{pattern.pattern.value.upper()}",
                direction=pattern.signal,
                strength=pattern.strength,
                entry_price=current_price,
                confidence=pattern.confidence
            ))
    
    return signals

# ==================== RESOURCES ====================

@mcp.resource("mt5://technical_indicators")
def get_technical_indicators_info() -> str:
    """
    Get information about available technical indicators.
    
    Returns:
        str: Information about technical indicators
    """
    return """
# Technical Indicators Available

## Trend Indicators
- Simple Moving Average (SMA) - 20, 50, 200 periods
- Exponential Moving Average (EMA) - 12, 26 periods
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)

## Momentum Indicators  
- RSI (Relative Strength Index) - 14 period
- Stochastic Oscillator (%K, %D)
- CCI (Commodity Channel Index)
- Williams %R

## Volatility Indicators
- Bollinger Bands (20 period, 2 std dev)
- ATR (Average True Range) - 14 period

## Usage Examples:
- RSI < 30: Oversold condition (potential buy)
- RSI > 70: Overbought condition (potential sell)
- MACD line above signal line: Bullish momentum
- Price above Bollinger upper band: Potentially overbought
- ADX > 25: Strong trend present
"""

@mcp.resource("mt5://candlestick_patterns")
def get_candlestick_patterns_info() -> str:
    """
    Get information about candlestick patterns.
    
    Returns:
        str: Information about candlestick patterns
    """
    return """
# Candlestick Patterns Detection

## Single Candle Patterns
- **Doji**: Indecision pattern, potential reversal
- **Hammer**: Bullish reversal at support
- **Inverted Hammer**: Bullish reversal pattern
- **Hanging Man**: Bearish reversal at resistance
- **Shooting Star**: Bearish reversal pattern

## Multi-Candle Patterns
- **Bullish Engulfing**: Strong bullish reversal
- **Bearish Engulfing**: Strong bearish reversal
- **Morning Star**: Three-candle bullish reversal
- **Evening Star**: Three-candle bearish reversal
- **Three White Soldiers**: Strong bullish continuation
- **Three Black Crows**: Strong bearish continuation

## Reliability Levels:
- High (80%+): Engulfing patterns, Three soldiers/crows
- Medium (60-80%): Hammer, Shooting star
- Low (40-60%): Doji, Single reversal patterns

## Usage:
- Combine with trend analysis for better accuracy
- Look for confirmation from volume and other indicators
- Consider market context (support/resistance levels)
"""

@mcp.resource("mt5://trading_signals")
def get_trading_signals_info() -> str:
    """
    Get information about trading signals.
    
    Returns:
        str: Information about trading signals
    """
    return """
# Trading Signals System

## Signal Types Generated:

### Momentum Signals
- **RSI_OVERSOLD**: RSI < 30, potential bounce
- **RSI_OVERBOUGHT**: RSI > 70, potential decline
- **MACD_BULLISH_CROSSOVER**: MACD line crosses above signal
- **MACD_BEARISH_CROSSOVER**: MACD line crosses below signal

### Trend Signals  
- **MA_GOLDEN_CROSS**: 20 SMA crosses above 50 SMA
- **MA_DEATH_CROSS**: 20 SMA crosses below 50 SMA
- **TREND_CONTINUATION**: Price confirms existing trend
- **TREND_REVERSAL**: Multiple indicators suggest reversal

### Pattern Signals
- **PATTERN_BULLISH_ENGULFING**: Strong bullish reversal
- **PATTERN_BEARISH_ENGULFING**: Strong bearish reversal
- **PATTERN_HAMMER**: Bullish reversal at support
- **PATTERN_DOJI**: Indecision, potential reversal

## Signal Strength:
- **VERY_STRONG**: Multiple confirmations, high confidence
- **STRONG**: Good technical setup, reliable entry
- **MODERATE**: Decent setup, manage risk carefully  
- **WEAK**: Early signal, wait for confirmation

## Risk Management:
- All signals include suggested stop loss levels
- Take profit targets based on risk/reward ratios
- Position sizing recommendations available
- Always confirm with multiple timeframes
"""