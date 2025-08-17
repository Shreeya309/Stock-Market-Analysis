import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import StockDataFetcher
from utils.technical_indicators import TechnicalIndicators

st.set_page_config(page_title="Technical Analysis", page_icon="📊", layout="wide")

st.title("📊 ML-Enhanced Technical Analysis")

# Initialize components
data_fetcher = StockDataFetcher()
tech_indicators = TechnicalIndicators()

# Sidebar controls
st.sidebar.header("📈 Analysis Settings")

# Stock selection
stock_symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'SPOT', 'ZOOM', 'SQ'
]

selected_stock = st.sidebar.selectbox(
    "Select Stock Symbol",
    stock_symbols,
    help="Choose a stock for technical analysis"
)

# Custom stock input
custom_stock = st.sidebar.text_input(
    "Or enter custom symbol:",
    placeholder="e.g., MSFT"
)

if custom_stock:
    selected_stock = custom_stock.upper()

# Analysis parameters
st.sidebar.subheader("🎯 Analysis Parameters")

data_period = st.sidebar.selectbox(
    "Data Period",
    ['6mo', '1y', '2y', '3y'],
    index=1,
    help="Historical data period for analysis"
)

chart_type = st.sidebar.selectbox(
    "Chart Type",
    ['Candlestick', 'Line', 'OHLC'],
    help="Select price chart visualization type"
)

# Technical indicators selection
st.sidebar.subheader("📊 Technical Indicators")

show_ma = st.sidebar.checkbox("Moving Averages", value=True)
show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_macd = st.sidebar.checkbox("MACD", value=True)
show_volume = st.sidebar.checkbox("Volume Analysis", value=True)

# Advanced indicators
with st.sidebar.expander("Advanced Indicators"):
    show_adx = st.checkbox("ADX (Trend Strength)")
    show_stochastic = st.checkbox("Stochastic Oscillator")
    show_williams = st.checkbox("Williams %R")
    show_cci = st.checkbox("Commodity Channel Index")
    show_obv = st.checkbox("On-Balance Volume")

# Pattern recognition
st.sidebar.subheader("🔍 Pattern Recognition")
enable_pattern_detection = st.sidebar.checkbox("Enable ML Pattern Detection", value=True)

# Main content
if st.button("🚀 Analyze Stock", type="primary"):
    
    with st.spinner(f"Analyzing {selected_stock}..."):
        try:
            # Fetch stock data
            stock_data = data_fetcher.fetch_stock_data(selected_stock, period=data_period)
            stock_info = data_fetcher.get_stock_info(selected_stock)
            
            if stock_data.empty:
                st.error(f"No data found for {selected_stock}. Please check the symbol.")
                st.stop()
            
            # Calculate technical indicators
            indicators_df = tech_indicators.calculate_all_indicators(stock_data)
            
            # Display stock info header
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = stock_data['Close'].iloc[-1]
                prev_close = stock_info.get('previous_close', stock_data['Close'].iloc[-2])
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{change_pct:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "Volume",
                    f"{stock_data['Volume'].iloc[-1]:,.0f}"
                )
            
            with col3:
                high_52 = stock_info.get('52_week_high', 0)
                low_52 = stock_info.get('52_week_low', 0)
                if high_52 > 0 and low_52 > 0:
                    position_52w = ((current_price - low_52) / (high_52 - low_52)) * 100
                    st.metric("52W Position", f"{position_52w:.1f}%")
                else:
                    st.metric("52W Position", "N/A")
            
            with col4:
                avg_volume = stock_info.get('avg_volume', 0)
                if avg_volume > 0:
                    volume_ratio = stock_data['Volume'].iloc[-1] / avg_volume
                    st.metric("Volume Ratio", f"{volume_ratio:.1f}x")
                else:
                    st.metric("Volume Ratio", "N/A")
            
            # Main price chart
            st.subheader("📈 Price Chart with Technical Indicators")
            
            # Create subplots
            rows = 1
            subplot_titles = ['Stock Price']
            
            if show_rsi or show_stochastic or show_williams:
                rows += 1
                subplot_titles.append('Oscillators')
            
            if show_macd:
                rows += 1
                subplot_titles.append('MACD')
            
            if show_volume:
                rows += 1
                subplot_titles.append('Volume')
            
            # Calculate row heights
            if rows == 1:
                row_heights = [1.0]
            elif rows == 2:
                row_heights = [0.7, 0.3]
            elif rows == 3:
                row_heights = [0.6, 0.2, 0.2]
            else:
                row_heights = [0.5, 0.2, 0.15, 0.15]
            
            fig = make_subplots(
                rows=rows, cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.05,
                row_heights=row_heights
            )
            
            # Price chart
            if chart_type == 'Candlestick':
                fig.add_trace(
                    go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name='Price'
                    ),
                    row=1, col=1
                )
            elif chart_type == 'OHLC':
                fig.add_trace(
                    go.Ohlc(
                        x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name='Price'
                    ),
                    row=1, col=1
                )
            else:  # Line chart
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
            
            # Moving Averages
            if show_ma:
                for ma_period in [20, 50, 200]:
                    if f'SMA_{ma_period}' in indicators_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=indicators_df.index,
                                y=indicators_df[f'SMA_{ma_period}'],
                                mode='lines',
                                name=f'SMA {ma_period}',
                                line=dict(width=1)
                            ),
                            row=1, col=1
                        )
            
            # Bollinger Bands
            if show_bollinger:
                fig.add_trace(
                    go.Scatter(
                        x=indicators_df.index,
                        y=indicators_df['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=indicators_df.index,
                        y=indicators_df['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            current_row = 1
            
            # Oscillators subplot
            if show_rsi or show_stochastic or show_williams:
                current_row += 1
                
                if show_rsi:
                    fig.add_trace(
                        go.Scatter(
                            x=indicators_df.index,
                            y=indicators_df['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ),
                        row=current_row, col=1
                    )
                    # RSI levels
                    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                                 annotation_text="Overbought", row=current_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                                 annotation_text="Oversold", row=current_row, col=1)
                
                if show_stochastic:
                    fig.add_trace(
                        go.Scatter(
                            x=indicators_df.index,
                            y=indicators_df['Stoch_K'],
                            mode='lines',
                            name='Stoch %K',
                            line=dict(color='orange', width=1)
                        ),
                        row=current_row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=indicators_df.index,
                            y=indicators_df['Stoch_D'],
                            mode='lines',
                            name='Stoch %D',
                            line=dict(color='red', width=1)
                        ),
                        row=current_row, col=1
                    )
                
                if show_williams:
                    fig.add_trace(
                        go.Scatter(
                            x=indicators_df.index,
                            y=indicators_df['Williams_R'],
                            mode='lines',
                            name='Williams %R',
                            line=dict(color='brown', width=1)
                        ),
                        row=current_row, col=1
                    )
            
            # MACD subplot
            if show_macd:
                current_row += 1
                fig.add_trace(
                    go.Scatter(
                        x=indicators_df.index,
                        y=indicators_df['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ),
                    row=current_row, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=indicators_df.index,
                        y=indicators_df['MACD_Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red', width=2)
                    ),
                    row=current_row, col=1
                )
                fig.add_trace(
                    go.Bar(
                        x=indicators_df.index,
                        y=indicators_df['MACD_Histogram'],
                        name='Histogram',
                        marker_color='green',
                        opacity=0.6
                    ),
                    row=current_row, col=1
                )
            
            # Volume subplot
            if show_volume:
                current_row += 1
                fig.add_trace(
                    go.Bar(
                        x=stock_data.index,
                        y=stock_data['Volume'],
                        name='Volume',
                        marker_color='lightblue',
                        opacity=0.7
                    ),
                    row=current_row, col=1
                )
                
                if show_obv:
                    fig.add_trace(
                        go.Scatter(
                            x=indicators_df.index,
                            y=indicators_df['OBV'],
                            mode='lines',
                            name='OBV',
                            line=dict(color='purple', width=2),
                            yaxis='y2'
                        ),
                        row=current_row, col=1
                    )
            
            fig.update_layout(
                title=f"{selected_stock} - Technical Analysis",
                height=800,
                showlegend=True,
                hovermode='x unified',
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current indicator values
            st.subheader("📋 Current Technical Indicator Values")
            
            latest_indicators = indicators_df.iloc[-1]
            
            # Create indicator cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**Trend Indicators**")
                sma_20 = latest_indicators['SMA_20']
                sma_50 = latest_indicators['SMA_50']
                trend = "Bullish" if current_price > sma_20 > sma_50 else "Bearish" if current_price < sma_20 < sma_50 else "Neutral"
                st.metric("Trend (SMA)", trend)
                st.metric("SMA 20", f"${sma_20:.2f}")
                st.metric("SMA 50", f"${sma_50:.2f}")
            
            with col2:
                st.markdown("**Momentum Indicators**")
                rsi = latest_indicators['RSI']
                rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                st.metric("RSI", f"{rsi:.1f}", rsi_signal)
                
                stoch_k = latest_indicators['Stoch_K']
                st.metric("Stochastic %K", f"{stoch_k:.1f}")
                
                williams_r = latest_indicators['Williams_R']
                st.metric("Williams %R", f"{williams_r:.1f}")
            
            with col3:
                st.markdown("**Volatility Indicators**")
                bb_position = latest_indicators['BB_Position']
                bb_signal = "Above Upper" if bb_position > 1 else "Below Lower" if bb_position < 0 else "In Bands"
                st.metric("BB Position", f"{bb_position:.2f}", bb_signal)
                
                atr = latest_indicators['ATR']
                st.metric("ATR", f"${atr:.2f}")
                
                volatility = latest_indicators['Price_Volatility'] * 100
                st.metric("Volatility", f"{volatility:.1f}%")
            
            with col4:
                st.markdown("**Volume Indicators**")
                if 'Volume_Ratio' in latest_indicators:
                    volume_ratio = latest_indicators['Volume_Ratio']
                    volume_signal = "High" if volume_ratio > 2 else "Low" if volume_ratio < 0.5 else "Normal"
                    st.metric("Volume Ratio", f"{volume_ratio:.1f}x", volume_signal)
                
                adx = latest_indicators['ADX']
                trend_strength = "Strong" if adx > 25 else "Weak" if adx < 20 else "Moderate"
                st.metric("ADX", f"{adx:.1f}", trend_strength)
                
                cci = latest_indicators['CCI']
                st.metric("CCI", f"{cci:.1f}")
            
            # Pattern detection
            if enable_pattern_detection:
                st.subheader("🔍 ML Pattern Detection")
                
                patterns_df = tech_indicators.detect_patterns(stock_data)
                signals_df = tech_indicators.generate_signals(stock_data)
                
                # Recent patterns
                recent_patterns = patterns_df.tail(20)
                pattern_summary = {}
                
                for col in recent_patterns.columns:
                    if recent_patterns[col].sum() > 0:
                        last_occurrence = recent_patterns[recent_patterns[col] == 1].index[-1]
                        pattern_summary[col] = last_occurrence
                
                if pattern_summary:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Recent Bullish Patterns**")
                        bullish_patterns = ['Golden_Cross', 'MACD_Bullish', 'Hammer', 'BB_Lower_Break']
                        for pattern in bullish_patterns:
                            if pattern in pattern_summary:
                                st.success(f"✅ {pattern.replace('_', ' ')}: {pattern_summary[pattern].strftime('%Y-%m-%d')}")
                    
                    with col2:
                        st.markdown("**Recent Bearish Patterns**")
                        bearish_patterns = ['Death_Cross', 'MACD_Bearish', 'BB_Upper_Break']
                        for pattern in bearish_patterns:
                            if pattern in pattern_summary:
                                st.error(f"❌ {pattern.replace('_', ' ')}: {pattern_summary[pattern].strftime('%Y-%m-%d')}")
                else:
                    st.info("No significant patterns detected in recent data.")
                
                # Trading signals
                st.subheader("🚦 Trading Signals")
                
                recent_signals = signals_df.tail(10)
                buy_signals = recent_signals[recent_signals['Buy_Signal'] == 1]
                sell_signals = recent_signals[recent_signals['Sell_Signal'] == 1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if len(buy_signals) > 0:
                        latest_buy = buy_signals.index[-1]
                        buy_strength = buy_signals['Buy_Strength'].iloc[-1]
                        st.success(f"🟢 Latest Buy Signal")
                        st.write(f"Date: {latest_buy.strftime('%Y-%m-%d')}")
                        st.write(f"Strength: {buy_strength}/4")
                    else:
                        st.info("No recent buy signals")
                
                with col2:
                    if len(sell_signals) > 0:
                        latest_sell = sell_signals.index[-1]
                        sell_strength = sell_signals['Sell_Strength'].iloc[-1]
                        st.error(f"🔴 Latest Sell Signal")
                        st.write(f"Date: {latest_sell.strftime('%Y-%m-%d')}")
                        st.write(f"Strength: {sell_strength}/4")
                    else:
                        st.info("No recent sell signals")
                
                with col3:
                    # Overall signal summary
                    current_signal = "Neutral"
                    if len(buy_signals) > 0 and len(sell_signals) > 0:
                        if buy_signals.index[-1] > sell_signals.index[-1]:
                            current_signal = "Buy"
                        else:
                            current_signal = "Sell"
                    elif len(buy_signals) > 0:
                        current_signal = "Buy"
                    elif len(sell_signals) > 0:
                        current_signal = "Sell"
                    
                    if current_signal == "Buy":
                        st.success(f"📈 Current Signal: {current_signal}")
                    elif current_signal == "Sell":
                        st.error(f"📉 Current Signal: {current_signal}")
                    else:
                        st.info(f"➡️ Current Signal: {current_signal}")
            
            # Support and resistance levels
            st.subheader("🎯 Support and Resistance Levels")
            
            support_resistance = tech_indicators.calculate_support_resistance(stock_data)
            latest_levels = support_resistance.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Support Levels**")
                st.metric("Support", f"${latest_levels['Support']:.2f}")
                st.metric("S1 (Pivot)", f"${latest_levels['S1']:.2f}")
                st.metric("S2 (Pivot)", f"${latest_levels['S2']:.2f}")
            
            with col2:
                st.markdown("**Current Price**")
                pivot = latest_levels['Pivot']
                distance_from_pivot = ((current_price - pivot) / pivot) * 100
                st.metric("Pivot Point", f"${pivot:.2f}")
                st.metric("Distance from Pivot", f"{distance_from_pivot:+.1f}%")
            
            with col3:
                st.markdown("**Resistance Levels**")
                st.metric("Resistance", f"${latest_levels['Resistance']:.2f}")
                st.metric("R1 (Pivot)", f"${latest_levels['R1']:.2f}")
                st.metric("R2 (Pivot)", f"${latest_levels['R2']:.2f}")
            
            # Advanced indicators section
            if show_adx or show_cci:
                st.subheader("📊 Advanced Technical Indicators")
                
                col1, col2 = st.columns(2)
                
                if show_adx:
                    with col1:
                        st.markdown("**ADX Analysis**")
                        adx_val = latest_indicators['ADX']
                        di_plus = latest_indicators['DI_Plus']
                        di_minus = latest_indicators['DI_Minus']
                        
                        trend_direction = "Bullish" if di_plus > di_minus else "Bearish"
                        st.metric("Trend Direction", trend_direction)
                        st.metric("DI+", f"{di_plus:.1f}")
                        st.metric("DI-", f"{di_minus:.1f}")
                
                if show_cci:
                    with col2:
                        st.markdown("**CCI Analysis**")
                        cci_val = latest_indicators['CCI']
                        cci_signal = "Overbought" if cci_val > 100 else "Oversold" if cci_val < -100 else "Normal"
                        st.metric("CCI Signal", cci_signal)
                        st.metric("CCI Value", f"{cci_val:.1f}")
            
        except Exception as e:
            st.error(f"Error performing technical analysis: {str(e)}")
            st.info("Please check the stock symbol and try again.")

# Educational section
with st.expander("📚 Understanding Technical Analysis"):
    st.markdown("""
    ### Key Technical Indicators Explained
    
    **Trend Indicators:**
    - **Moving Averages (SMA/EMA)**: Show the average price over a specific period, helping identify trend direction
    - **ADX**: Measures trend strength (>25 = strong trend, <20 = weak trend)
    
    **Momentum Oscillators:**
    - **RSI**: Measures overbought (>70) and oversold (<30) conditions
    - **Stochastic**: Compares closing price to price range over time
    - **Williams %R**: Similar to Stochastic but inverted scale
    
    **Volatility Indicators:**
    - **Bollinger Bands**: Price channels based on standard deviation
    - **ATR**: Measures price volatility regardless of direction
    
    **Volume Indicators:**
    - **OBV**: Combines price and volume to show buying/selling pressure
    - **Volume Ratio**: Compares current volume to average volume
    
    ### Pattern Recognition
    Our ML algorithms detect common chart patterns and generate trading signals based on multiple indicators.
    
    ### Important Notes
    - Technical analysis should be combined with fundamental analysis
    - No single indicator is 100% accurate
    - Always use proper risk management
    """)

# Footer
st.markdown("---")
st.markdown("*Technical analysis is for educational purposes only and should not be considered as financial advice.*")
