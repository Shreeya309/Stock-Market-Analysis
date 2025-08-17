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

from models.lstm_model import LSTMStockPredictor
from models.rf_model import RandomForestStockPredictor
from utils.data_fetcher import StockDataFetcher
from utils.technical_indicators import TechnicalIndicators

st.set_page_config(page_title="Stock Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Stock Price Prediction")

# Initialize components
data_fetcher = StockDataFetcher()
tech_indicators = TechnicalIndicators()

# Sidebar controls
st.sidebar.header("📊 Prediction Settings")

# Stock selection
stock_symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'SPOT', 'ZOOM', 'SQ'
]

selected_stock = st.sidebar.selectbox(
    "Select Stock Symbol",
    stock_symbols,
    help="Choose a stock for prediction analysis"
)

# Custom stock input
custom_stock = st.sidebar.text_input(
    "Or enter custom symbol:",
    placeholder="e.g., MSFT"
)

if custom_stock:
    selected_stock = custom_stock.upper()

# Prediction parameters
st.sidebar.subheader("🎯 Prediction Parameters")

prediction_days = st.sidebar.slider(
    "Days to Predict",
    min_value=1,
    max_value=60,
    value=30,
    help="Number of days to predict into the future"
)

data_period = st.sidebar.selectbox(
    "Historical Data Period",
    ['1y', '2y', '3y', '5y'],
    index=1,
    help="Amount of historical data to use for training"
)

model_type = st.sidebar.selectbox(
    "Model Type",
    ['LSTM', 'Random Forest', 'Both'],
    help="Choose which ML model(s) to use for prediction"
)

# Advanced settings
st.sidebar.subheader("⚙️ Advanced Settings")

with st.sidebar.expander("LSTM Settings"):
    lstm_epochs = st.slider("Training Epochs", 10, 100, 50)
    lstm_sequence_length = st.slider("Sequence Length", 30, 120, 60)
    lstm_units = st.slider("LSTM Units", 25, 100, 50)

with st.sidebar.expander("Random Forest Settings"):
    rf_n_estimators = st.slider("Number of Trees", 50, 500, 100)
    rf_max_depth = st.slider("Max Depth", 5, 30, 10)

# Main content
if st.button("🚀 Generate Predictions", type="primary"):
    
    with st.spinner(f"Fetching data for {selected_stock}..."):
        try:
            # Fetch stock data
            stock_data = data_fetcher.fetch_stock_data(selected_stock, period=data_period)
            stock_info = data_fetcher.get_stock_info(selected_stock)
            
            if stock_data.empty:
                st.error(f"No data found for {selected_stock}. Please check the symbol.")
                st.stop()
            
            # Display stock info
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
                    "Market Cap",
                    f"${stock_info.get('market_cap', 0) / 1e9:.1f}B" if stock_info.get('market_cap', 0) > 0 else "N/A"
                )
            
            with col3:
                st.metric(
                    "P/E Ratio",
                    f"{stock_info.get('pe_ratio', 0):.2f}" if stock_info.get('pe_ratio', 0) > 0 else "N/A"
                )
            
            with col4:
                st.metric(
                    "Beta",
                    f"{stock_info.get('beta', 0):.2f}" if stock_info.get('beta', 0) > 0 else "N/A"
                )
            
            # Model training and predictions
            predictions_results = {}
            
            if model_type in ['LSTM', 'Both']:
                st.subheader("🧠 LSTM Neural Network Prediction")
                
                with st.spinner("Training LSTM model..."):
                    lstm_model = LSTMStockPredictor(
                        sequence_length=lstm_sequence_length,
                        units=lstm_units,
                        dropout=0.2
                    )
                    
                    # Train model
                    metrics, training_data = lstm_model.train(
                        stock_data,
                        epochs=lstm_epochs,
                        verbose=0
                    )
                    
                    # Generate predictions
                    lstm_predictions = lstm_model.predict_future(stock_data, days=prediction_days)
                    predictions_results['LSTM'] = {
                        'predictions': lstm_predictions,
                        'metrics': metrics,
                        'model': lstm_model
                    }
                
                # Display LSTM metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training MSE", f"{metrics['train_mse']:.6f}")
                with col2:
                    st.metric("Test MSE", f"{metrics['test_mse']:.6f}")
                with col3:
                    st.metric("Test MAE", f"{metrics['test_mae']:.6f}")
            
            if model_type in ['Random Forest', 'Both']:
                st.subheader("🌲 Random Forest Prediction")
                
                with st.spinner("Training Random Forest model..."):
                    rf_model = RandomForestStockPredictor(
                        n_estimators=rf_n_estimators,
                        max_depth=rf_max_depth
                    )
                    
                    # Train model
                    rf_metrics, rf_training_data = rf_model.train(stock_data)
                    
                    # Generate predictions
                    rf_predictions = rf_model.predict_future(stock_data, days=prediction_days)
                    predictions_results['Random Forest'] = {
                        'predictions': rf_predictions,
                        'metrics': rf_metrics,
                        'model': rf_model
                    }
                
                # Display RF metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training R²", f"{rf_metrics['train_r2']:.4f}")
                with col2:
                    st.metric("Test R²", f"{rf_metrics['test_r2']:.4f}")
                with col3:
                    st.metric("Test MAE", f"{rf_metrics['test_mae']:.4f}")
            
            # Visualization
            st.subheader("📈 Prediction Visualization")
            
            # Create prediction chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Stock Price & Predictions', 'Volume'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Historical prices
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index[-252:],  # Last year
                    y=stock_data['Close'].iloc[-252:],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add predictions
            colors = ['red', 'green', 'orange']
            for i, (model_name, results) in enumerate(predictions_results.items()):
                predictions = results['predictions']
                fig.add_trace(
                    go.Scatter(
                        x=predictions.index,
                        y=predictions.values,
                        mode='lines+markers',
                        name=f'{model_name} Prediction',
                        line=dict(color=colors[i], width=2, dash='dash'),
                        marker=dict(size=4)
                    ),
                    row=1, col=1
                )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=stock_data.index[-252:],
                    y=stock_data['Volume'].iloc[-252:],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.6
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f"{selected_stock} - Price Prediction Analysis",
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction summary
            st.subheader("📋 Prediction Summary")
            
            current_price = stock_data['Close'].iloc[-1]
            summary_data = []
            
            for model_name, results in predictions_results.items():
                predictions = results['predictions']
                
                # Calculate prediction statistics
                final_price = predictions.iloc[-1]
                price_change = final_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                # Prediction direction
                direction = "📈 Bullish" if price_change > 0 else "📉 Bearish"
                
                summary_data.append({
                    'Model': model_name,
                    'Current Price': f"${current_price:.2f}",
                    'Predicted Price': f"${final_price:.2f}",
                    'Price Change': f"${price_change:+.2f}",
                    'Change %': f"{price_change_pct:+.2f}%",
                    'Direction': direction,
                    'Confidence': "High" if abs(price_change_pct) > 5 else "Medium" if abs(price_change_pct) > 2 else "Low"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Feature importance (for Random Forest)
            if 'Random Forest' in predictions_results:
                st.subheader("🎯 Feature Importance (Random Forest)")
                
                importance_df = predictions_results['Random Forest']['model'].get_feature_importance()
                
                fig_importance = px.bar(
                    importance_df.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Most Important Features",
                    labels={'importance': 'Feature Importance', 'feature': 'Technical Indicators'}
                )
                
                fig_importance.update_layout(height=500)
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Technical analysis
            st.subheader("🔍 Technical Analysis")
            
            # Calculate technical indicators
            indicators_df = tech_indicators.calculate_all_indicators(stock_data)
            patterns_df = tech_indicators.detect_patterns(stock_data)
            signals_df = tech_indicators.generate_signals(stock_data)
            
            # Current technical indicators
            latest_indicators = indicators_df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rsi = latest_indicators['RSI']
                rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                st.metric("RSI", f"{rsi:.1f}", rsi_signal)
            
            with col2:
                macd = latest_indicators['MACD']
                macd_signal = latest_indicators['MACD_Signal']
                macd_status = "Bullish" if macd > macd_signal else "Bearish"
                st.metric("MACD", f"{macd:.4f}", macd_status)
            
            with col3:
                bb_position = latest_indicators['BB_Position']
                bb_signal = "Above Upper" if bb_position > 1 else "Below Lower" if bb_position < 0 else "In Range"
                st.metric("BB Position", f"{bb_position:.2f}", bb_signal)
            
            with col4:
                adx = latest_indicators['ADX']
                trend_strength = "Strong" if adx > 25 else "Weak" if adx < 20 else "Moderate"
                st.metric("ADX", f"{adx:.1f}", trend_strength)
            
            # Recent signals
            recent_signals = signals_df.tail(10)
            buy_signals = recent_signals[recent_signals['Buy_Signal'] == 1]
            sell_signals = recent_signals[recent_signals['Sell_Signal'] == 1]
            
            if len(buy_signals) > 0 or len(sell_signals) > 0:
                st.subheader("🚦 Recent Trading Signals")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(buy_signals) > 0:
                        st.success(f"🟢 Latest Buy Signal: {buy_signals.index[-1].strftime('%Y-%m-%d')}")
                        st.info(f"Signal Strength: {buy_signals['Buy_Strength'].iloc[-1]}/4")
                
                with col2:
                    if len(sell_signals) > 0:
                        st.error(f"🔴 Latest Sell Signal: {sell_signals.index[-1].strftime('%Y-%m-%d')}")
                        st.info(f"Signal Strength: {sell_signals['Sell_Strength'].iloc[-1]}/4")
            
            # Risk assessment
            st.subheader("⚠️ Risk Assessment")
            
            returns = stock_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility as percentage
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                var_95 = np.percentile(returns, 5) * 100
                st.metric("1-Day VaR (95%)", f"{var_95:.2f}%")
            
            with col2:
                st.metric("Annual Volatility", f"{volatility:.1f}%")
            
            with col3:
                sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            # Model comparison (if both models used)
            if len(predictions_results) > 1:
                st.subheader("🏆 Model Comparison")
                
                comparison_data = []
                for model_name, results in predictions_results.items():
                    predictions = results['predictions']
                    final_price = predictions.iloc[-1]
                    price_change_pct = ((final_price - current_price) / current_price) * 100
                    
                    # Model-specific metrics
                    if model_name == 'LSTM':
                        accuracy_metric = 1 - results['metrics']['test_mse']  # Simplified accuracy
                    else:  # Random Forest
                        accuracy_metric = results['metrics']['test_r2']
                    
                    comparison_data.append({
                        'Model': model_name,
                        'Predicted Return': f"{price_change_pct:+.2f}%",
                        'Model Accuracy': f"{accuracy_metric:.4f}",
                        'Prediction Confidence': "High" if abs(price_change_pct) > 5 else "Medium"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            st.info("Please check the stock symbol and try again.")

# Educational section
with st.expander("📚 Understanding Stock Prediction Models"):
    st.markdown("""
    ### LSTM (Long Short-Term Memory) Neural Networks
    - **Best for**: Capturing complex temporal patterns and long-term dependencies
    - **Strengths**: Excellent for sequential data, can learn from historical patterns
    - **Considerations**: Requires more data and computation time, can overfit with small datasets
    
    ### Random Forest
    - **Best for**: Feature-rich predictions using technical indicators
    - **Strengths**: Robust, interpretable, handles mixed data types well
    - **Considerations**: May not capture complex temporal relationships as well as LSTM
    
    ### Key Metrics
    - **MSE (Mean Squared Error)**: Lower values indicate better prediction accuracy
    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
    - **R² Score**: Proportion of variance explained by the model (higher is better)
    
    ### Important Notes
    - Predictions are based on historical patterns and may not account for future market events
    - Always combine with fundamental analysis and risk management
    - Past performance does not guarantee future results
    """)

# Footer
st.markdown("---")
st.markdown("*Predictions are for educational purposes only and should not be considered as financial advice.*")
