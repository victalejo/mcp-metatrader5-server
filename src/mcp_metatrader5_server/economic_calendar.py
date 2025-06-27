"""
MetaTrader 5 MCP Server - Economic Calendar Module

This module provides comprehensive economic calendar functionality including:
- Real-time economic events from RapidAPI
- Market impact analysis
- Event filtering and categorization
- Trading alerts and notifications
- Currency impact predictions
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta, date
from enum import Enum
import requests
from dataclasses import dataclass

import pandas as pd
import numpy as np
from fastmcp import FastMCP
from pydantic import BaseModel, Field, validator

# Import the main server instance
from mcp_metatrader5_server.server import mcp

logger = logging.getLogger("mt5-mcp-server.economic_calendar")

# Configuration
RAPIDAPI_KEY = "eec4f771a6msh8457d8adae76424p1d4793jsn5c6ff732d2b8"
RAPIDAPI_HOST = "ultimate-economic-calendar.p.rapidapi.com"
BASE_URL = "https://ultimate-economic-calendar.p.rapidapi.com/economic-events/tradingview"

# Enums
class EventImportance(str, Enum):
    LOW = "low"          # importance = -1
    MEDIUM = "medium"    # importance = 0  
    HIGH = "high"        # importance = 1
    CRITICAL = "critical" # importance > 1

class MarketImpact(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"

class CurrencyCode(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    AUD = "AUD"
    CAD = "CAD"
    CHF = "CHF"
    NZD = "NZD"

# Pydantic Models
class EconomicEvent(BaseModel):
    """Economic event model based on API response"""
    id: str
    title: str
    indicator: str
    country: str
    currency: str
    date: datetime
    period: str
    importance: int
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    unit: str
    scale: str = ""
    source: str = ""
    comment: str = ""
    link: str = ""

    @validator('date', pre=True)
    def parse_date(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    @property
    def importance_level(self) -> EventImportance:
        """Get importance level as enum"""
        if self.importance <= -1:
            return EventImportance.LOW
        elif self.importance == 0:
            return EventImportance.MEDIUM
        elif self.importance == 1:
            return EventImportance.HIGH
        else:
            return EventImportance.CRITICAL

    @property
    def surprise_factor(self) -> Optional[float]:
        """Calculate surprise factor if actual vs forecast available"""
        if self.actual is not None and self.forecast is not None and self.forecast != 0:
            return ((self.actual - self.forecast) / abs(self.forecast)) * 100
        return None

    @property
    def momentum_factor(self) -> Optional[float]:
        """Calculate momentum factor if actual vs previous available"""
        if self.actual is not None and self.previous is not None and self.previous != 0:
            return ((self.actual - self.previous) / abs(self.previous)) * 100
        return None

class EventFilter(BaseModel):
    """Filter criteria for economic events"""
    countries: Optional[List[str]] = None
    currencies: Optional[List[str]] = None
    importance_min: Optional[int] = None
    indicators: Optional[List[str]] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    only_with_forecast: bool = False
    only_high_impact: bool = False

class MarketImpactAnalysis(BaseModel):
    """Market impact analysis result"""
    event: EconomicEvent
    expected_impact: MarketImpact
    impact_magnitude: float  # 0-100 scale
    affected_pairs: List[str]
    trading_recommendation: str
    confidence_score: float
    time_horizon: str  # immediate, short_term, medium_term

class CalendarSummary(BaseModel):
    """Daily calendar summary"""
    date: date
    total_events: int
    high_impact_events: int
    major_releases: List[str]
    currency_focus: List[str]
    market_sentiment: str
    volatility_expectation: str

# ==================== API FUNCTIONS ====================

def _get_api_headers() -> Dict[str, str]:
    """Get API headers for RapidAPI requests"""
    return {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }

@mcp.tool()
def fetch_economic_events(
    date_from: str,
    date_to: str,
    countries: Optional[str] = "US,DE,GB,JP,AU,CA,CH,NZ"
) -> List[EconomicEvent]:
    """
    Fetch economic events from RapidAPI calendar.
    
    Args:
        date_from: Start date in YYYY-MM-DD format
        date_to: End date in YYYY-MM-DD format
        countries: Comma-separated country codes (default: major economies)
        
    Returns:
        List[EconomicEvent]: List of economic events
    """
    try:
        querystring = {
            "from": date_from,
            "to": date_to,
            "countries": countries
        }
        
        headers = _get_api_headers()
        
        response = requests.get(BASE_URL, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") != "ok":
            logger.error(f"API returned error status: {data}")
            return []
        
        events = []
        for event_data in data.get("result", []):
            try:
                event = EconomicEvent(**event_data)
                events.append(event)
            except Exception as e:
                logger.warning(f"Failed to parse event: {event_data}, error: {e}")
                continue
        
        logger.info(f"Fetched {len(events)} economic events from {date_from} to {date_to}")
        return events
        
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching economic events: {e}")
        return []

@mcp.tool()
def get_today_events(countries: Optional[str] = "US,DE,GB,JP,AU,CA,CH,NZ") -> List[EconomicEvent]:
    """
    Get today's economic events.
    
    Args:
        countries: Comma-separated country codes
        
    Returns:
        List[EconomicEvent]: Today's economic events
    """
    today = datetime.now().date()
    return fetch_economic_events(
        date_from=today.isoformat(),
        date_to=today.isoformat(),
        countries=countries
    )

@mcp.tool()
def get_week_events(countries: Optional[str] = "US,DE,GB,JP,AU,CA,CH,NZ") -> List[EconomicEvent]:
    """
    Get this week's economic events.
    
    Args:
        countries: Comma-separated country codes
        
    Returns:
        List[EconomicEvent]: This week's economic events
    """
    today = datetime.now().date()
    week_end = today + timedelta(days=7-today.weekday())
    
    return fetch_economic_events(
        date_from=today.isoformat(),
        date_to=week_end.isoformat(),
        countries=countries
    )

@mcp.tool()
def get_high_impact_events(
    date_from: str,
    date_to: str,
    countries: Optional[str] = "US,DE,GB,JP,AU,CA,CH,NZ"
) -> List[EconomicEvent]:
    """
    Get high impact economic events only.
    
    Args:
        date_from: Start date in YYYY-MM-DD format
        date_to: End date in YYYY-MM-DD format
        countries: Comma-separated country codes
        
    Returns:
        List[EconomicEvent]: High impact events only
    """
    all_events = fetch_economic_events(date_from, date_to, countries)
    return [event for event in all_events if event.importance >= 1]

# ==================== ANALYSIS FUNCTIONS ====================

@mcp.tool()
def filter_events(events: List[EconomicEvent], filter_criteria: EventFilter) -> List[EconomicEvent]:
    """
    Filter economic events based on criteria.
    
    Args:
        events: List of economic events
        filter_criteria: Filter criteria
        
    Returns:
        List[EconomicEvent]: Filtered events
    """
    filtered_events = events.copy()
    
    if filter_criteria.countries:
        filtered_events = [e for e in filtered_events if e.country in filter_criteria.countries]
    
    if filter_criteria.currencies:
        filtered_events = [e for e in filtered_events if e.currency in filter_criteria.currencies]
    
    if filter_criteria.importance_min is not None:
        filtered_events = [e for e in filtered_events if e.importance >= filter_criteria.importance_min]
    
    if filter_criteria.indicators:
        filtered_events = [e for e in filtered_events if e.indicator in filter_criteria.indicators]
    
    if filter_criteria.date_from:
        filtered_events = [e for e in filtered_events if e.date.date() >= filter_criteria.date_from]
    
    if filter_criteria.date_to:
        filtered_events = [e for e in filtered_events if e.date.date() <= filter_criteria.date_to]
    
    if filter_criteria.only_with_forecast:
        filtered_events = [e for e in filtered_events if e.forecast is not None]
    
    if filter_criteria.only_high_impact:
        filtered_events = [e for e in filtered_events if e.importance >= 1]
    
    return filtered_events

@mcp.tool()
def analyze_market_impact(event: EconomicEvent) -> MarketImpactAnalysis:
    """
    Analyze potential market impact of an economic event.
    
    Args:
        event: Economic event to analyze
        
    Returns:
        MarketImpactAnalysis: Market impact analysis
    """
    # Determine affected currency pairs
    affected_pairs = []
    major_pairs = {
        "USD": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF"],
        "EUR": ["EURUSD", "EURGBP", "EURJPY", "EURCAD", "EURAUD", "EURNZD", "EURCHF"],
        "GBP": ["GBPUSD", "EURGBP", "GBPJPY", "GBPCAD", "GBPAUD", "GBPNZD", "GBPCHF"],
        "JPY": ["USDJPY", "EURGBP", "GBPJPY", "CADJPY", "AUDJPY", "NZDJPY", "CHFJPY"],
        "AUD": ["AUDUSD", "EURAUD", "GBPAUD", "AUDJPY", "AUDCAD", "AUDNZD", "AUDCHF"],
        "CAD": ["USDCAD", "EURCAD", "GBPCAD", "CADJPY", "AUDCAD", "CADCHF"],
        "CHF": ["USDCHF", "EURCHF", "GBPCHF", "CHFJPY", "AUDCHF", "CADCHF"],
        "NZD": ["NZDUSD", "EURNZD", "GBPNZD", "NZDJPY", "AUDNZD", "NZDCAD", "NZDCHF"]
    }
    
    affected_pairs = major_pairs.get(event.currency, [])
    
    # Calculate impact magnitude
    base_impact = min(max(event.importance, 0), 3) * 25  # 0-75 base impact
    
    # Add surprise factor
    surprise_bonus = 0
    if event.surprise_factor is not None:
        surprise_bonus = min(abs(event.surprise_factor), 25)  # Up to 25 points
    
    impact_magnitude = min(base_impact + surprise_bonus, 100)
    
    # Determine expected impact direction
    expected_impact = MarketImpact.NEUTRAL
    if event.surprise_factor is not None:
        if abs(event.surprise_factor) > 10:  # Significant surprise
            if event.surprise_factor > 0:
                expected_impact = MarketImpact.BULLISH if _is_positive_indicator(event.indicator) else MarketImpact.BEARISH
            else:
                expected_impact = MarketImpact.BEARISH if _is_positive_indicator(event.indicator) else MarketImpact.BULLISH
        
        if event.importance >= 1 and abs(event.surprise_factor) > 20:
            expected_impact = MarketImpact.VOLATILE
    
    # Generate trading recommendation
    trading_recommendation = _generate_trading_recommendation(event, expected_impact, impact_magnitude)
    
    # Calculate confidence score
    confidence_score = _calculate_confidence_score(event, impact_magnitude)
    
    # Determine time horizon
    time_horizon = "immediate" if event.importance >= 1 else "short_term"
    if event.indicator in ["GDP", "Employment", "Inflation", "Interest Rate"]:
        time_horizon = "medium_term"
    
    return MarketImpactAnalysis(
        event=event,
        expected_impact=expected_impact,
        impact_magnitude=impact_magnitude,
        affected_pairs=affected_pairs,
        trading_recommendation=trading_recommendation,
        confidence_score=confidence_score,
        time_horizon=time_horizon
    )

def _is_positive_indicator(indicator: str) -> bool:
    """Determine if indicator is generally positive for currency"""
    positive_indicators = [
        "GDP", "Employment", "Retail Sales", "Industrial Production",
        "Manufacturing", "Consumer Confidence", "Business Confidence"
    ]
    negative_indicators = [
        "Unemployment", "Inflation", "Trade Deficit", "Budget Deficit"
    ]
    
    indicator_lower = indicator.lower()
    
    for pos in positive_indicators:
        if pos.lower() in indicator_lower:
            return True
    
    for neg in negative_indicators:
        if neg.lower() in indicator_lower:
            return False
    
    return True  # Default to positive

def _generate_trading_recommendation(event: EconomicEvent, impact: MarketImpact, magnitude: float) -> str:
    """Generate trading recommendation based on analysis"""
    if magnitude < 30:
        return "Monitor event but low trading impact expected"
    
    if impact == MarketImpact.BULLISH:
        return f"Consider long {event.currency} positions. Expected bullish impact."
    elif impact == MarketImpact.BEARISH:
        return f"Consider short {event.currency} positions. Expected bearish impact."
    elif impact == MarketImpact.VOLATILE:
        return f"High volatility expected for {event.currency}. Use tight stops and consider straddle strategies."
    else:
        return "Neutral impact expected. Monitor for unexpected movements."

def _calculate_confidence_score(event: EconomicEvent, magnitude: float) -> float:
    """Calculate confidence score for the analysis"""
    base_confidence = 0.5
    
    # Higher importance = higher confidence
    base_confidence += (event.importance / 4) * 0.3
    
    # Having forecast increases confidence
    if event.forecast is not None:
        base_confidence += 0.1
    
    # Having previous value increases confidence
    if event.previous is not None:
        base_confidence += 0.1
    
    # Higher magnitude = higher confidence (up to a point)
    magnitude_factor = min(magnitude / 100, 0.8) * 0.2
    base_confidence += magnitude_factor
    
    return min(base_confidence, 1.0)

@mcp.tool()
def create_daily_summary(events: List[EconomicEvent], target_date: date) -> CalendarSummary:
    """
    Create a daily calendar summary.
    
    Args:
        events: List of economic events
        target_date: Date to summarize
        
    Returns:
        CalendarSummary: Daily summary
    """
    # Filter events for the target date
    day_events = [e for e in events if e.date.date() == target_date]
    
    # Count high impact events
    high_impact_count = len([e for e in day_events if e.importance >= 1])
    
    # Identify major releases
    major_releases = []
    important_indicators = ["GDP", "Employment", "Inflation", "Interest Rate", "NFP", "CPI", "Retail Sales"]
    
    for event in day_events:
        if event.importance >= 1:
            for indicator in important_indicators:
                if indicator.lower() in event.indicator.lower():
                    major_releases.append(f"{event.country} {event.indicator}")
                    break
    
    # Determine currency focus
    currency_counts = {}
    for event in day_events:
        if event.importance >= 0:  # Medium impact or higher
            currency_counts[event.currency] = currency_counts.get(event.currency, 0) + 1
    
    currency_focus = sorted(currency_counts.keys(), key=lambda x: currency_counts[x], reverse=True)[:3]
    
    # Determine market sentiment
    if high_impact_count == 0:
        market_sentiment = "Calm"
        volatility_expectation = "Low"
    elif high_impact_count <= 2:
        market_sentiment = "Moderate Activity"
        volatility_expectation = "Moderate"
    else:
        market_sentiment = "High Activity"
        volatility_expectation = "High"
    
    return CalendarSummary(
        date=target_date,
        total_events=len(day_events),
        high_impact_events=high_impact_count,
        major_releases=major_releases,
        currency_focus=currency_focus,
        market_sentiment=market_sentiment,
        volatility_expectation=volatility_expectation
    )

@mcp.tool()
def get_currency_specific_events(
    currency: str,
    date_from: str,
    date_to: str
) -> List[EconomicEvent]:
    """
    Get events specific to a currency.
    
    Args:
        currency: Currency code (e.g., "USD", "EUR")
        date_from: Start date in YYYY-MM-DD format
        date_to: End date in YYYY-MM-DD format
        
    Returns:
        List[EconomicEvent]: Currency-specific events
    """
    all_events = fetch_economic_events(date_from, date_to)
    return [event for event in all_events if event.currency == currency.upper()]

@mcp.tool()
def find_events_by_indicator(
    indicator_name: str,
    date_from: str,
    date_to: str
) -> List[EconomicEvent]:
    """
    Find events by specific indicator name.
    
    Args:
        indicator_name: Name of the economic indicator
        date_from: Start date in YYYY-MM-DD format
        date_to: End date in YYYY-MM-DD format
        
    Returns:
        List[EconomicEvent]: Events matching the indicator
    """
    all_events = fetch_economic_events(date_from, date_to)
    return [event for event in all_events if indicator_name.lower() in event.indicator.lower()]

@mcp.tool()
def get_surprise_events(
    date_from: str,
    date_to: str,
    min_surprise: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Get events with significant surprises (actual vs forecast).
    
    Args:
        date_from: Start date in YYYY-MM-DD format
        date_to: End date in YYYY-MM-DD format
        min_surprise: Minimum surprise percentage
        
    Returns:
        List[Dict[str, Any]]: Events with surprises and their analysis
    """
    all_events = fetch_economic_events(date_from, date_to)
    surprise_events = []
    
    for event in all_events:
        if event.surprise_factor is not None and abs(event.surprise_factor) >= min_surprise:
            analysis = analyze_market_impact(event)
            surprise_events.append({
                "event": event.dict(),
                "surprise_factor": event.surprise_factor,
                "impact_analysis": analysis.dict()
            })
    
    # Sort by surprise magnitude
    surprise_events.sort(key=lambda x: abs(x["surprise_factor"]), reverse=True)
    
    return surprise_events

# ==================== INTEGRATION WITH TRADING ====================

@mcp.tool()
def correlate_events_with_market(
    symbol: str,
    events: List[EconomicEvent],
    market_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Correlate economic events with market movements.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        events: List of economic events
        market_data: Market price data
        
    Returns:
        Dict[str, Any]: Correlation analysis
    """
    if not market_data or not events:
        return {"correlation": 0, "significant_events": [], "analysis": "Insufficient data"}
    
    # Convert market data to DataFrame
    df = pd.DataFrame(market_data)
    df['time'] = pd.to_datetime(df['time'])
    
    # Analyze price movements around event times
    significant_events = []
    
    for event in events:
        if event.importance < 1:  # Only analyze high impact events
            continue
        
        event_time = event.date
        
        # Find market data around event time (±2 hours)
        before_time = event_time - timedelta(hours=2)
        after_time = event_time + timedelta(hours=2)
        
        before_data = df[df['time'] <= event_time]
        after_data = df[df['time'] >= event_time]
        
        if len(before_data) > 0 and len(after_data) > 0:
            price_before = before_data['close'].iloc[-1]
            price_after = after_data['close'].iloc[0] if len(after_data) > 0 else price_before
            
            price_change = ((price_after - price_before) / price_before) * 100
            
            if abs(price_change) > 0.1:  # Significant movement (>0.1%)
                significant_events.append({
                    "event": event.dict(),
                    "price_change": price_change,
                    "volatility_increase": abs(price_change) > 0.5
                })
    
    # Calculate overall correlation
    correlation_score = len(significant_events) / max(len([e for e in events if e.importance >= 1]), 1)
    
    return {
        "symbol": symbol,
        "correlation_score": correlation_score,
        "significant_events": significant_events,
        "total_events_analyzed": len([e for e in events if e.importance >= 1]),
        "events_with_impact": len(significant_events),
        "analysis": f"Found {len(significant_events)} events with significant market impact"
    }

# ==================== RESOURCES ====================

@mcp.resource("mt5://economic_calendar")
def get_economic_calendar_info() -> str:
    """
    Get information about economic calendar functionality.
    
    Returns:
        str: Information about economic calendar
    """
    return """
# Economic Calendar - Guía Completa

## 📅 Funcionalidades Disponibles

### Obtención de Eventos
- `fetch_economic_events()`: Obtener eventos por rango de fechas
- `get_today_events()`: Eventos de hoy
- `get_week_events()`: Eventos de la semana
- `get_high_impact_events()`: Solo eventos de alto impacto

### Análisis de Impacto
- `analyze_market_impact()`: Análisis completo de impacto en mercado
- `get_surprise_events()`: Eventos con sorpresas significativas
- `correlate_events_with_market()`: Correlación con movimientos de precios

### Filtros y Búsqueda
- `filter_events()`: Filtrar por múltiples criterios
- `get_currency_specific_events()`: Eventos por moneda
- `find_events_by_indicator()`: Buscar por indicador específico

## 🎯 Niveles de Importancia

### Importancia Baja (-1)
- Eventos de bajo impacto
- Generalmente no mueven mercados
- Ejemplo: Datos agrícolas menores

### Importancia Media (0)
- Impacto moderado en mercados
- Puede causar volatilidad menor
- Ejemplo: Índices regionales

### Importancia Alta (1)
- Alto impacto en mercados
- Puede causar movimientos significativos
- Ejemplo: NFP, CPI, GDP

### Importancia Crítica (>1)
- Impacto extremo en mercados
- Movimientos mayores esperados
- Ejemplo: Decisiones de tasas de interés

## 💱 Impacto por Monedas

### USD (Dólar Estadounidense)
- **Indicadores clave**: NFP, CPI, GDP, Fed Decisions
- **Pares afectados**: Todos los pares USD
- **Horarios**: Principalmente 12:30-16:00 GMT

### EUR (Euro)
- **Indicadores clave**: ECB Decisions, German GDP, EU CPI
- **Pares afectados**: EUR/USD, EUR/GBP, EUR/JPY
- **Horarios**: Principalmente 07:00-11:00 GMT

### GBP (Libra Esterlina)
- **Indicadores clave**: BoE Decisions, UK GDP, Employment
- **Pares afectados**: GBP/USD, EUR/GBP, GBP/JPY
- **Horarios**: Principalmente 08:30-10:30 GMT

## 📊 Análisis de Sorpresas

### Factor Sorpresa = (Actual - Pronóstico) / |Pronóstico| × 100

- **>20%**: Sorpresa muy significativa
- **10-20%**: Sorpresa significativa
- **5-10%**: Sorpresa moderada
- **<5%**: Sorpresa menor

## 🚨 Recomendaciones de Trading

### Antes del Evento
1. Revisar importancia y pronósticos
2. Identificar pares potencialmente afectados
3. Preparar estrategias de entrada/salida
4. Configurar alertas de volatilidad

### Durante el Evento
1. Monitorear liberación en tiempo real
2. Comparar actual vs pronóstico
3. Evaluar reacción inicial del mercado
4. Ajustar posiciones según análisis

### Después del Evento
1. Analizar movimientos de precios
2. Evaluar si la reacción fue apropiada
3. Buscar oportunidades de continuación
4. Documentar correlaciones para futuro

## ⚠️ Gestión de Riesgos

### Eventos de Alto Impacto
- Reducir tamaños de posición
- Usar stops más amplios
- Considerar cerrar posiciones antes del evento
- Preparar para alta volatilidad

### Estrategias Recomendadas
- **Straddle**: Para alta volatilidad esperada
- **Breakout**: Seguir movimientos post-evento
- **Fade**: Contra-tendencia si reacción exagerada
- **News Trading**: Entrada rápida post-liberación
"""

@mcp.resource("mt5://major_economic_indicators")
def get_major_indicators_info() -> str:
    """
    Get information about major economic indicators.
    
    Returns:
        str: Information about major economic indicators
    """
    return """
# Indicadores Económicos Principales

## 🇺🇸 Estados Unidos

### Non-Farm Payrolls (NFP)
- **Liberación**: Primer viernes del mes, 12:30 GMT
- **Impacto**: Muy Alto
- **Descripción**: Empleos creados fuera del sector agrícola
- **Trading**: Movimientos de 50-150 pips en USD común

### Consumer Price Index (CPI)
- **Liberación**: Segundo martes del mes, 12:30 GMT
- **Impacto**: Muy Alto
- **Descripción**: Medida principal de inflación
- **Trading**: Afecta expectativas de Fed, volatilidad alta

### Federal Reserve Decisions
- **Liberación**: 8 veces al año, 18:00 GMT
- **Impacto**: Extremo
- **Descripción**: Decisiones de política monetaria
- **Trading**: Puede mover mercados 200+ pips

### GDP (Gross Domestic Product)
- **Liberación**: Trimestral, 12:30 GMT
- **Impacto**: Alto
- **Descripción**: Crecimiento económico total
- **Trading**: Confirma tendencias económicas

## 🇪🇺 Zona Euro

### ECB Interest Rate Decision
- **Liberación**: 8 veces al año, 11:45 GMT
- **Impacto**: Extremo
- **Descripción**: Decisiones del Banco Central Europeo
- **Trading**: Mayor impacto en EUR

### Eurozone CPI
- **Liberación**: Mensual, 09:00 GMT
- **Impacto**: Alto
- **Descripción**: Inflación de la Zona Euro
- **Trading**: Afecta política del ECB

### German GDP
- **Liberación**: Trimestral, 06:00 GMT
- **Impacto**: Alto
- **Descripción**: Economía más grande de Europa
- **Trading**: Proxy para salud de Eurozona

## 🇬🇧 Reino Unido

### BoE Interest Rate Decision
- **Liberación**: 8 veces al año, 11:00 GMT
- **Impacto**: Extremo
- **Descripción**: Banco de Inglaterra decisiones
- **Trading**: Impacto directo en GBP

### UK Employment Data
- **Liberación**: Mensual, 08:30 GMT
- **Impacto**: Alto
- **Descripción**: Datos de empleo y desempleo
- **Trading**: Afecta perspectivas económicas

### UK Retail Sales
- **Liberación**: Mensual, 08:30 GMT
- **Impacto**: Medio-Alto
- **Descripción**: Gastos del consumidor
- **Trading**: Indicador de salud económica

## 🇯🇵 Japón

### BoJ Interest Rate Decision
- **Liberación**: 8 veces al año, 03:00 GMT
- **Impacto**: Extremo
- **Descripción**: Banco de Japón decisiones
- **Trading**: Impacto principal en JPY

### Japan GDP
- **Liberación**: Trimestral, 23:50 GMT
- **Impacto**: Alto
- **Descripción**: Crecimiento económico japonés
- **Trading**: Tendencias a largo plazo

### Tankan Survey
- **Liberación**: Trimestral, 23:50 GMT
- **Impacto**: Alto
- **Descripción**: Confianza empresarial
- **Trading**: Sentimiento del mercado japonés

## 📈 Calendario de Liberaciones

### Primera Semana del Mes
- NFP (Viernes)
- ISM Manufacturing (Lunes)
- Factory Orders (Martes)

### Segunda Semana del Mes
- CPI USA (Martes)
- PPI USA (Miércoles)
- Retail Sales (Jueves)

### Tercera Semana del Mes
- Housing Starts (Martes)
- FOMC Minutes (Miércoles)
- Leading Indicators (Jueves)

### Cuarta Semana del Mes
- Durable Goods (Miércoles)
- GDP Preliminary (Jueves)
- Consumer Sentiment (Viernes)

## 🎯 Estrategias por Indicador

### Para NFP
- Straddle antes de liberación
- Breakout en dirección de sorpresa
- Seguimiento de tendencia post-evento

### Para Decisiones de Bancos Centrales
- Posiciones largas antes de decisión
- Monitorear conferencias de prensa
- Tradear volatilidad implícita

### Para Datos de Inflación
- Correlación con bonos gubernamentales
- Impacto en expectativas de tasas
- Movimientos de monedas a largo plazo
"""

