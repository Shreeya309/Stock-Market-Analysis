import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.title("📈 ML Model Performance Analysis")

# Initialize components
data_fetcher = StockDataFetcher()
tech_indicators = TechnicalIndicators()

# Sidebar controls
st.sidebar.header("🎯 Performance Analysis Settings")

# Model testing configuration
st.sidebar.subheader("🔬 Model Testing")

# Stock selection for testing
test_stocks = st.sidebar.multiselect(
    "Select Stocks for Testing",
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
     'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'JPM', 'BAC', 'KO'],
    default=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
    help="Choose stocks for model performance evaluation"
)

# Performance metrics to analyze
st.sidebar.subheader("📊 Performance Metrics")

metrics_to_show = st.sidebar.multiselect(
    "Select Metrics to Display",
    ['MSE', 'MAE', 'RMSE', 'R²', 'MAPE', 'Direction Accuracy'],
    default=['MSE', 'MAE', 'R²', 'Direction Accuracy'],
    help="Choose which performance metrics to calculate"
)

# Model parameters
st.sidebar.subheader("⚙️ Model Parameters")

data_period = st.sidebar.selectbox(
    "Training Data Period",
    ['1y', '2y', '3y'],
    index=1,
    help="Amount of data for model training"
)

test_size = st.sidebar.slider(
    "Test Set Size (%)",
    min_value=10,
    max_value=40,
    value=20,
    help="Percentage of data reserved for testing"
) / 100

prediction_horizons = st.sidebar.multiselect(
    "Prediction Horizons (days)",
    [1, 5, 10, 20, 30],
    default=[1, 5, 10],
    help="Number of days ahead to predict"
)

# Advanced model settings
with st.sidebar.expander("🔧 Advanced Model Settings"):
    lstm_epochs = st.slider("LSTM Epochs", 20, 100, 50)
    lstm_sequence_length = st.slider("LSTM Sequence Length", 30, 120, 60)
    rf_n_estimators = st.slider("RF Estimators", 50, 300, 100)
    cross_validation_folds = st.slider("Cross Validation Folds", 3, 10, 5)

# Comparison settings
st.sidebar.subheader("🔄 Model Comparison")
enable_baseline = st.sidebar.checkbox("Include Baseline Models", value=True)
enable_ensemble = st.sidebar.checkbox("Test Ensemble Methods", value=True)

# Main content
if st.button("🚀 Run Performance Analysis", type="primary"):
    
    if not test_stocks:
        st.error("Please select at least one stock for performance testing.")
        st.stop()
    
    if not prediction_horizons:
        st.error("Please select at least one prediction horizon.")
        st.stop()
    
    with st.spinner("Running comprehensive model performance analysis..."):
        try:
            # Initialize results storage
            all_results = {}
            model_comparison = {}
            
            # Fetch data for all test stocks
            st.info(f"Fetching data for {len(test_stocks)} stocks...")
            
            progress_bar = st.progress(0)
            
            for idx, stock in enumerate(test_stocks):
                try:
                    st.write(f"Analyzing {stock}...")
                    
                    # Fetch stock data
                    stock_data = data_fetcher.fetch_stock_data(stock, period=data_period)
                    
                    if stock_data.empty:
                        st.warning(f"No data available for {stock}")
                        continue
                    
                    stock_results = {}
                    
                    # Test different prediction horizons
                    for horizon in prediction_horizons:
                        horizon_results = {}
                        
                        # LSTM Model
                        try:
                            lstm_model = LSTMStockPredictor(
                                sequence_length=lstm_sequence_length,
                                units=50,
                                dropout=0.2
                            )
                            
                            # Train LSTM
                            lstm_metrics, lstm_data = lstm_model.train(
                                stock_data,
                                test_size=test_size,
                                epochs=lstm_epochs,
                                verbose=0
                            )
                            
                            # Calculate additional metrics for LSTM
                            X_train, X_test, y_train, y_test, train_pred, test_pred = lstm_data
                            
                            lstm_performance = {
                                'MSE': lstm_metrics['test_mse'],
                                'MAE': lstm_metrics['test_mae'],
                                'RMSE': np.sqrt(lstm_metrics['test_mse']),
                                'R²': 1 - (lstm_metrics['test_mse'] / np.var(y_test)),
                                'MAPE': np.mean(np.abs((y_test - test_pred) / y_test)) * 100,
                                'Direction_Accuracy': calculate_direction_accuracy(y_test, test_pred),
                                'predictions': test_pred,
                                'actual': y_test
                            }
                            
                            horizon_results['LSTM'] = lstm_performance
                            
                        except Exception as e:
                            st.warning(f"LSTM failed for {stock} (horizon {horizon}): {str(e)}")
                        
                        # Random Forest Model
                        try:
                            rf_model = RandomForestStockPredictor(
                                n_estimators=rf_n_estimators,
                                max_depth=10
                            )
                            
                            # Train Random Forest
                            rf_metrics, rf_data = rf_model.train(stock_data, test_size=test_size)
                            
                            X_train, X_test, y_train, y_test, train_pred, test_pred = rf_data
                            
                            rf_performance = {
                                'MSE': rf_metrics['test_mse'],
                                'MAE': rf_metrics['test_mae'],
                                'RMSE': np.sqrt(rf_metrics['test_mse']),
                                'R²': rf_metrics['test_r2'],
                                'MAPE': np.mean(np.abs((y_test - test_pred) / y_test)) * 100,
                                'Direction_Accuracy': calculate_direction_accuracy(y_test, test_pred),
                                'predictions': test_pred,
                                'actual': y_test
                            }
                            
                            horizon_results['Random Forest'] = rf_performance
                            
                        except Exception as e:
                            st.warning(f"Random Forest failed for {stock} (horizon {horizon}): {str(e)}")
                        
                        # Baseline models (if enabled)
                        if enable_baseline:
                            try:
                                # Simple moving average baseline
                                returns = stock_data['Close'].pct_change().dropna()
                                
                                # Split for baseline
                                train_size = int(len(returns) * (1 - test_size))
                                test_returns = returns.iloc[train_size:]
                                
                                # Random walk baseline (predict no change)
                                baseline_pred = np.zeros(len(test_returns))
                                actual_returns = test_returns.values
                                
                                baseline_performance = {
                                    'MSE': mean_squared_error(actual_returns, baseline_pred),
                                    'MAE': mean_absolute_error(actual_returns, baseline_pred),
                                    'RMSE': np.sqrt(mean_squared_error(actual_returns, baseline_pred)),
                                    'R²': r2_score(actual_returns, baseline_pred),
                                    'MAPE': np.mean(np.abs(actual_returns)) * 100,
                                    'Direction_Accuracy': 0.5,  # Random walk assumption
                                    'predictions': baseline_pred,
                                    'actual': actual_returns
                                }
                                
                                horizon_results['Baseline (Random Walk)'] = baseline_performance
                                
                            except Exception as e:
                                st.warning(f"Baseline failed for {stock}: {str(e)}")
                        
                        stock_results[f'{horizon}d'] = horizon_results
                    
                    all_results[stock] = stock_results
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(test_stocks))
                    
                except Exception as e:
                    st.error(f"Failed to analyze {stock}: {str(e)}")
            
            progress_bar.empty()
            
            if not all_results:
                st.error("No results generated. Please check your settings and try again.")
                st.stop()
            
            # Display results
            st.subheader("📊 Model Performance Summary")
            
            # Create performance comparison table
            comparison_data = []
            
            for stock in all_results.keys():
                for horizon in all_results[stock].keys():
                    for model in all_results[stock][horizon].keys():
                        metrics = all_results[stock][horizon][model]
                        
                        row = {
                            'Stock': stock,
                            'Horizon': horizon,
                            'Model': model
                        }
                        
                        for metric in metrics_to_show:
                            if metric == 'Direction Accuracy':
                                row[metric] = f"{metrics.get('Direction_Accuracy', 0):.1%}"
                            elif metric == 'MAPE':
                                row[metric] = f"{metrics.get('MAPE', 0):.2f}%"
                            elif metric == 'R²':
                                row[metric] = f"{metrics.get('R²', 0):.4f}"
                            else:
                                row[metric] = f"{metrics.get(metric, 0):.6f}"
                        
                        comparison_data.append(row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Model performance visualization
                st.subheader("📈 Performance Visualization")
                
                # Create performance charts for each metric
                for metric in metrics_to_show:
                    if metric in ['MSE', 'MAE', 'RMSE', 'MAPE']:  # Lower is better
                        chart_title = f"{metric} Comparison (Lower is Better)"
                    else:  # Higher is better
                        chart_title = f"{metric} Comparison (Higher is Better)"
                    
                    # Create grouped bar chart
                    metric_data = []
                    for _, row in comparison_df.iterrows():
                        value_str = row[metric]
                        # Extract numeric value
                        if '%' in value_str:
                            value = float(value_str.replace('%', ''))
                        else:
                            value = float(value_str)
                        
                        metric_data.append({
                            'Stock': row['Stock'],
                            'Model': row['Model'],
                            'Horizon': row['Horizon'],
                            'Value': value
                        })
                    
                    metric_df = pd.DataFrame(metric_data)
                    
                    if not metric_df.empty:
                        # Group by horizon and create subplots
                        unique_horizons = sorted(metric_df['Horizon'].unique())
                        
                        fig = make_subplots(
                            rows=1, cols=len(unique_horizons),
                            subplot_titles=[f"Horizon: {h}" for h in unique_horizons]
                        )
                        
                        colors = px.colors.qualitative.Set1
                        model_colors = {}
                        
                        for col_idx, horizon in enumerate(unique_horizons):
                            horizon_data = metric_df[metric_df['Horizon'] == horizon]
                            
                            for model_idx, model in enumerate(horizon_data['Model'].unique()):
                                if model not in model_colors:
                                    model_colors[model] = colors[len(model_colors) % len(colors)]
                                
                                model_data = horizon_data[horizon_data['Model'] == model]
                                
                                fig.add_trace(
                                    go.Bar(
                                        x=model_data['Stock'],
                                        y=model_data['Value'],
                                        name=model,
                                        marker_color=model_colors[model],
                                        showlegend=(col_idx == 0)  # Only show legend for first subplot
                                    ),
                                    row=1, col=col_idx + 1
                                )
                        
                        fig.update_layout(
                            title=chart_title,
                            height=500,
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Average performance across all stocks
                st.subheader("🏆 Average Model Performance")
                
                avg_performance = {}
                
                for horizon in prediction_horizons:
                    horizon_str = f'{horizon}d'
                    avg_performance[horizon_str] = {}
                    
                    # Get all models tested
                    all_models = set()
                    for stock in all_results.keys():
                        if horizon_str in all_results[stock]:
                            all_models.update(all_results[stock][horizon_str].keys())
                    
                    for model in all_models:
                        model_metrics = {}
                        
                        for metric in metrics_to_show:
                            values = []
                            for stock in all_results.keys():
                                if (horizon_str in all_results[stock] and 
                                    model in all_results[stock][horizon_str]):
                                    
                                    metric_key = metric.replace(' ', '_') if metric == 'Direction Accuracy' else metric
                                    value = all_results[stock][horizon_str][model].get(metric_key, 0)
                                    values.append(value)
                            
                            if values:
                                model_metrics[metric] = np.mean(values)
                        
                        avg_performance[horizon_str][model] = model_metrics
                
                # Display average performance table
                avg_data = []
                for horizon in avg_performance.keys():
                    for model in avg_performance[horizon].keys():
                        row = {
                            'Horizon': horizon,
                            'Model': model
                        }
                        
                        for metric in metrics_to_show:
                            value = avg_performance[horizon][model].get(metric, 0)
                            
                            if metric == 'Direction Accuracy':
                                row[metric] = f"{value:.1%}"
                            elif metric == 'MAPE':
                                row[metric] = f"{value:.2f}%"
                            elif metric == 'R²':
                                row[metric] = f"{value:.4f}"
                            else:
                                row[metric] = f"{value:.6f}"
                        
                        avg_data.append(row)
                
                avg_df = pd.DataFrame(avg_data)
                st.dataframe(avg_df, use_container_width=True)
                
                # Model ranking
                st.subheader("🥇 Model Rankings")
                
                # Rank models by R² score (if available)
                if 'R²' in metrics_to_show:
                    rankings = {}
                    
                    for horizon in avg_performance.keys():
                        horizon_ranking = []
                        
                        for model in avg_performance[horizon].keys():
                            r2_score = avg_performance[horizon][model].get('R²', 0)
                            horizon_ranking.append((model, r2_score))
                        
                        horizon_ranking.sort(key=lambda x: x[1], reverse=True)
                        rankings[horizon] = horizon_ranking
                    
                    # Display rankings
                    for horizon, ranking in rankings.items():
                        st.write(f"**{horizon} Prediction Ranking (by R² Score):**")
                        
                        for rank, (model, score) in enumerate(ranking, 1):
                            if rank == 1:
                                st.success(f"🥇 {rank}. {model}: {score:.4f}")
                            elif rank == 2:
                                st.info(f"🥈 {rank}. {model}: {score:.4f}")
                            elif rank == 3:
                                st.warning(f"🥉 {rank}. {model}: {score:.4f}")
                            else:
                                st.write(f"{rank}. {model}: {score:.4f}")
                
                # Prediction accuracy over time
                st.subheader("📅 Prediction Accuracy Over Time")
                
                # Show prediction vs actual for best performing stock/model combination
                if len(test_stocks) > 0 and len(prediction_horizons) > 0:
                    # Find best performing model
                    best_stock = test_stocks[0]
                    best_horizon = f'{prediction_horizons[0]}d'
                    best_model = None
                    best_r2 = -np.inf
                    
                    if (best_stock in all_results and 
                        best_horizon in all_results[best_stock]):
                        
                        for model in all_results[best_stock][best_horizon].keys():
                            r2 = all_results[best_stock][best_horizon][model].get('R²', -np.inf)
                            if r2 > best_r2:
                                best_r2 = r2
                                best_model = model
                    
                    if best_model:
                        predictions = all_results[best_stock][best_horizon][best_model]['predictions']
                        actual = all_results[best_stock][best_horizon][best_model]['actual']
                        
                        fig_pred = go.Figure()
                        
                        # Create time index for plotting
                        time_index = list(range(len(actual)))
                        
                        fig_pred.add_trace(
                            go.Scatter(
                                x=time_index,
                                y=actual,
                                mode='lines',
                                name='Actual',
                                line=dict(color='blue', width=2)
                            )
                        )
                        
                        fig_pred.add_trace(
                            go.Scatter(
                                x=time_index,
                                y=predictions,
                                mode='lines',
                                name='Predicted',
                                line=dict(color='red', width=2, dash='dash')
                            )
                        )
                        
                        fig_pred.update_layout(
                            title=f"Best Model Performance: {best_model} on {best_stock} ({best_horizon})",
                            xaxis_title="Time Step",
                            yaxis_title="Price",
                            height=400
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Scatter plot of predictions vs actual
                        fig_scatter = go.Figure()
                        
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=actual,
                                y=predictions,
                                mode='markers',
                                name='Predictions',
                                marker=dict(color='blue', size=6, opacity=0.6)
                            )
                        )
                        
                        # Add perfect prediction line
                        min_val = min(np.min(actual), np.min(predictions))
                        max_val = max(np.max(actual), np.max(predictions))
                        
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            )
                        )
                        
                        fig_scatter.update_layout(
                            title="Predictions vs Actual Values",
                            xaxis_title="Actual Values",
                            yaxis_title="Predicted Values",
                            height=400
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
            else:
                st.warning("No performance data to display.")
                
        except Exception as e:
            st.error(f"Error during performance analysis: {str(e)}")
            st.info("Please check your settings and try again.")

def calculate_direction_accuracy(actual, predicted):
    """
    Calculate directional accuracy (percentage of correct up/down predictions)
    """
    if len(actual) <= 1 or len(predicted) <= 1:
        return 0.0
    
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    
    return np.mean(actual_direction == predicted_direction)

# Educational section
with st.expander("📚 Understanding Model Performance Metrics"):
    st.markdown("""
    ### Performance Metrics Explained
    
    **Regression Metrics:**
    - **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values (lower is better)
    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values (lower is better)
    - **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target variable (lower is better)
    - **R² (R-squared)**: Proportion of variance explained by the model (higher is better, max = 1)
    - **MAPE (Mean Absolute Percentage Error)**: Average percentage error (lower is better)
    
    **Trading-Specific Metrics:**
    - **Direction Accuracy**: Percentage of correct up/down price movement predictions
    - **Sharpe Ratio**: Risk-adjusted return measure for trading strategies
    
    ### Model Comparison Guidelines
    
    **LSTM vs Random Forest:**
    - **LSTM**: Better for capturing temporal patterns and long-term dependencies
    - **Random Forest**: Better for feature-rich environments with technical indicators
    
    **Baseline Models:**
    - **Random Walk**: Assumes no predictability (predicts no change)
    - **Moving Average**: Simple trend-following approach
    
    ### Important Considerations
    - **Overfitting**: High training performance but poor test performance
    - **Data Leakage**: Using future information to predict past values
    - **Market Regime Changes**: Models may perform differently in different market conditions
    - **Transaction Costs**: Real trading performance may differ due to costs
    
    ### Best Practices
    - Use walk-forward analysis for time series data
    - Combine multiple metrics for comprehensive evaluation
    - Test on out-of-sample data periods
    - Consider practical trading constraints
    """)

# Footer
st.markdown("---")
st.markdown("*Model performance analysis is for educational purposes only and should not be considered as financial advice.*")
