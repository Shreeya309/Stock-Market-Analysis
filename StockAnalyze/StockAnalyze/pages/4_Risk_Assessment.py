import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import StockDataFetcher
from utils.risk_metrics import RiskMetrics

st.set_page_config(page_title="Risk Assessment", page_icon="⚠️", layout="wide")

st.title("⚠️ Comprehensive Risk Assessment")

# Initialize components
data_fetcher = StockDataFetcher()
risk_calculator = RiskMetrics()

# Sidebar controls
st.sidebar.header("🎯 Risk Analysis Settings")

# Asset selection
st.sidebar.subheader("📊 Asset Selection")

analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ['Single Asset', 'Portfolio', 'Comparative Analysis'],
    help="Choose the type of risk analysis"
)

if analysis_type == 'Single Asset':
    # Single stock analysis
    stock_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'SPOT', 'ZOOM', 'SQ'
    ]
    
    selected_stock = st.sidebar.selectbox(
        "Select Stock Symbol",
        stock_symbols,
        help="Choose a stock for risk analysis"
    )
    
    # Custom stock input
    custom_stock = st.sidebar.text_input(
        "Or enter custom symbol:",
        placeholder="e.g., MSFT"
    )
    
    if custom_stock:
        selected_stock = custom_stock.upper()
        selected_assets = [selected_stock]
    else:
        selected_assets = [selected_stock]

elif analysis_type == 'Portfolio':
    # Portfolio analysis
    st.sidebar.write("Enter portfolio weights (symbol:weight):")
    portfolio_input = st.sidebar.text_area(
        "Portfolio Composition",
        placeholder="AAPL:0.3\nMSFT:0.25\nGOOGL:0.2\nAMZN:0.15\nTSLA:0.1",
        height=100
    )
    
    if portfolio_input:
        portfolio_weights = {}
        for line in portfolio_input.split('\n'):
            if ':' in line:
                symbol, weight = line.strip().split(':')
                try:
                    portfolio_weights[symbol.upper()] = float(weight)
                except:
                    pass
        selected_assets = list(portfolio_weights.keys())
    else:
        # Default portfolio
        portfolio_weights = {'AAPL': 0.3, 'MSFT': 0.25, 'GOOGL': 0.2, 'AMZN': 0.15, 'TSLA': 0.1}
        selected_assets = list(portfolio_weights.keys())

else:  # Comparative Analysis
    # Multiple assets comparison
    comparison_assets = st.sidebar.multiselect(
        "Select Assets to Compare",
        ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
         'SPY', 'QQQ', 'VTI', 'BTC-USD', 'ETH-USD'],
        default=['AAPL', 'MSFT', 'TSLA', 'SPY']
    )
    selected_assets = comparison_assets

# Risk analysis parameters
st.sidebar.subheader("📈 Analysis Parameters")

data_period = st.sidebar.selectbox(
    "Data Period",
    ['1y', '2y', '3y', '5y'],
    index=1,
    help="Historical data period for risk calculation"
)

confidence_levels = st.sidebar.multiselect(
    "VaR Confidence Levels",
    [90, 95, 99],
    default=[95, 99],
    help="Confidence levels for Value at Risk calculation"
)

var_method = st.sidebar.selectbox(
    "VaR Calculation Method",
    ['Historical', 'Parametric', 'Monte Carlo'],
    help="Method for calculating Value at Risk"
)

# Benchmark selection
benchmark_symbol = st.sidebar.selectbox(
    "Benchmark",
    ['^GSPC', '^IXIC', '^DJI', 'SPY', 'QQQ', 'VTI'],
    index=0,
    help="Benchmark for relative risk metrics"
)

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    monte_carlo_sims = st.slider("Monte Carlo Simulations", 1000, 50000, 10000)
    rolling_window = st.slider("Rolling Window (days)", 30, 252, 126)
    stress_test_enabled = st.checkbox("Enable Stress Testing", value=True)

# Main content
if st.button("🚀 Analyze Risk", type="primary"):
    
    if not selected_assets:
        st.error("Please select at least one asset for risk analysis.")
        st.stop()
    
    with st.spinner("Calculating risk metrics..."):
        try:
            # Fetch data
            asset_data = {}
            for symbol in selected_assets:
                try:
                    data = data_fetcher.fetch_stock_data(symbol, period=data_period)
                    if not data.empty:
                        asset_data[symbol] = data
                except Exception as e:
                    st.warning(f"Could not fetch data for {symbol}: {str(e)}")
            
            if not asset_data:
                st.error("No data could be fetched for the selected assets.")
                st.stop()
            
            # Fetch benchmark data
            benchmark_data = data_fetcher.fetch_stock_data(benchmark_symbol, period=data_period)
            
            # Display asset overview
            st.subheader("📊 Asset Overview")
            
            overview_data = []
            for symbol, data in asset_data.items():
                current_price = data['Close'].iloc[-1]
                returns = data['Close'].pct_change().dropna()
                
                overview_data.append({
                    'Symbol': symbol,
                    'Current Price': f"${current_price:.2f}",
                    'Daily Returns (Ann.)': f"{returns.mean() * 252:.1%}",
                    'Volatility (Ann.)': f"{returns.std() * np.sqrt(252):.1%}",
                    'Sharpe Ratio': f"{risk_calculator.sharpe_ratio(returns):.3f}",
                    'Max Drawdown': f"{risk_calculator.maximum_drawdown(data['Close'])['max_drawdown']:.1%}"
                })
            
            overview_df = pd.DataFrame(overview_data)
            st.dataframe(overview_df, use_container_width=True)
            
            if analysis_type == 'Single Asset':
                # Single asset risk analysis
                symbol = selected_assets[0]
                asset_prices = asset_data[symbol]['Close']
                asset_returns = asset_prices.pct_change().dropna()
                
                st.subheader(f"🎯 {symbol} - Detailed Risk Analysis")
                
                # Calculate comprehensive risk metrics
                risk_summary = risk_calculator.risk_metrics_summary(
                    asset_prices,
                    benchmark_prices=benchmark_data['Close'] if not benchmark_data.empty else None
                )
                
                # Risk metrics display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Return Metrics**")
                    st.metric("Total Return", f"{risk_summary['total_return']:.1%}")
                    st.metric("Annual Return", f"{risk_summary['annual_return']:.1%}")
                    st.metric("Sharpe Ratio", f"{risk_summary['sharpe_ratio']:.3f}")
                
                with col2:
                    st.markdown("**Risk Metrics**")
                    st.metric("Annual Volatility", f"{risk_summary['annual_volatility']:.1%}")
                    st.metric("VaR (95%)", f"{risk_summary['var_95']:.2%}")
                    st.metric("CVaR (95%)", f"{risk_summary['cvar_95']:.2%}")
                
                with col3:
                    st.markdown("**Drawdown Metrics**")
                    st.metric("Max Drawdown", f"{risk_summary['max_drawdown']:.1%}")
                    st.metric("Ulcer Index", f"{risk_summary['ulcer_index']:.4f}")
                    st.metric("Calmar Ratio", f"{risk_summary['calmar_ratio']:.3f}")
                
                with col4:
                    st.markdown("**Distribution Metrics**")
                    st.metric("Skewness", f"{risk_summary['skewness']:.3f}")
                    st.metric("Kurtosis", f"{risk_summary['kurtosis']:.3f}")
                    st.metric("Win Rate", f"{risk_summary['win_rate']:.1%}")
                
                # Price and drawdown chart
                st.subheader("📈 Price Performance and Drawdown")
                
                fig_price = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['Price Performance', 'Drawdown'],
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )
                
                # Normalize price to 100 at start
                normalized_price = (asset_prices / asset_prices.iloc[0]) * 100
                
                fig_price.add_trace(
                    go.Scatter(
                        x=normalized_price.index,
                        y=normalized_price,
                        mode='lines',
                        name='Price Performance',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                # Drawdown
                dd_data = risk_calculator.maximum_drawdown(asset_prices)
                
                fig_price.add_trace(
                    go.Scatter(
                        x=dd_data['drawdown_series'].index,
                        y=dd_data['drawdown_series'] * 100,
                        mode='lines',
                        name='Drawdown',
                        fill='tonexty',
                        line=dict(color='red'),
                        fillcolor='rgba(255,0,0,0.3)'
                    ),
                    row=2, col=1
                )
                
                fig_price.update_layout(
                    title=f"{symbol} - Risk Analysis",
                    height=500,
                    showlegend=True
                )
                
                fig_price.update_yaxes(title_text="Normalized Price", row=1, col=1)
                fig_price.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
                fig_price.update_xaxes(title_text="Date", row=2, col=1)
                
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Rolling risk metrics
                st.subheader("📊 Rolling Risk Metrics")
                
                # Calculate rolling metrics
                rolling_vol = asset_returns.rolling(window=rolling_window).std() * np.sqrt(252) * 100
                rolling_sharpe = asset_returns.rolling(window=rolling_window).apply(
                    lambda x: risk_calculator.sharpe_ratio(x)
                )
                
                fig_rolling = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['Rolling Volatility (Annualized)', 'Rolling Sharpe Ratio'],
                    vertical_spacing=0.1
                )
                
                fig_rolling.add_trace(
                    go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol,
                        mode='lines',
                        name='Rolling Volatility',
                        line=dict(color='orange', width=2)
                    ),
                    row=1, col=1
                )
                
                fig_rolling.add_trace(
                    go.Scatter(
                        x=rolling_sharpe.index,
                        y=rolling_sharpe,
                        mode='lines',
                        name='Rolling Sharpe',
                        line=dict(color='green', width=2)
                    ),
                    row=2, col=1
                )
                
                fig_rolling.update_layout(
                    title=f"Rolling Risk Metrics ({rolling_window} days)",
                    height=500
                )
                
                fig_rolling.update_yaxes(title_text="Volatility (%)", row=1, col=1)
                fig_rolling.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
                fig_rolling.update_xaxes(title_text="Date", row=2, col=1)
                
                st.plotly_chart(fig_rolling, use_container_width=True)
                
                # Return distribution analysis
                st.subheader("📊 Return Distribution Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig_hist = go.Figure()
                    
                    returns_pct = asset_returns * 100
                    
                    fig_hist.add_trace(
                        go.Histogram(
                            x=returns_pct,
                            nbinsx=50,
                            name='Daily Returns',
                            opacity=0.7
                        )
                    )
                    
                    # Add VaR lines
                    for cl in confidence_levels:
                        var_value = risk_calculator.value_at_risk(asset_returns, cl/100, var_method.lower()) * 100
                        fig_hist.add_vline(
                            x=var_value,
                            line_dash="dash",
                            annotation_text=f"VaR {cl}%: {var_value:.2f}%"
                        )
                    
                    fig_hist.update_layout(
                        title="Daily Returns Distribution",
                        xaxis_title="Daily Returns (%)",
                        yaxis_title="Frequency"
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Q-Q plot for normality test
                    fig_qq = go.Figure()
                    
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(asset_returns)))
                    sample_quantiles = np.sort(asset_returns)
                    
                    fig_qq.add_trace(
                        go.Scatter(
                            x=theoretical_quantiles,
                            y=sample_quantiles,
                            mode='markers',
                            name='Sample vs Normal',
                            marker=dict(color='blue', size=4)
                        )
                    )
                    
                    # Add reference line
                    fig_qq.add_trace(
                        go.Scatter(
                            x=theoretical_quantiles,
                            y=theoretical_quantiles * asset_returns.std() + asset_returns.mean(),
                            mode='lines',
                            name='Normal Distribution',
                            line=dict(color='red', dash='dash')
                        )
                    )
                    
                    fig_qq.update_layout(
                        title="Q-Q Plot (Normality Test)",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles"
                    )
                    
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                # Monte Carlo simulation
                st.subheader("🎲 Monte Carlo Risk Simulation")
                
                mc_results = risk_calculator.monte_carlo_simulation(
                    asset_prices,
                    num_simulations=monte_carlo_sims,
                    days=252
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Expected Return (1Y)", f"{(mc_results['expected_final_value'] - 1):.1%}")
                
                with col2:
                    st.metric("95% VaR (1Y)", f"{mc_results['var_95']:.1%}")
                
                with col3:
                    st.metric("Probability of Loss", f"{mc_results['probability_of_loss']:.1%}")
                
                # Monte Carlo paths visualization
                fig_mc = go.Figure()
                
                # Sample paths (show subset for performance)
                num_paths_show = min(100, mc_results['simulated_paths'].shape[1])
                sample_indices = np.random.choice(
                    mc_results['simulated_paths'].shape[1],
                    num_paths_show,
                    replace=False
                )
                
                for i in sample_indices:
                    fig_mc.add_trace(
                        go.Scatter(
                            x=list(range(252)),
                            y=mc_results['simulated_paths'][:, i],
                            mode='lines',
                            line=dict(width=0.5, color='lightblue'),
                            showlegend=False,
                            hovertemplate=None,
                            hoverinfo='skip'
                        )
                    )
                
                # Average path
                avg_path = np.mean(mc_results['simulated_paths'], axis=1)
                fig_mc.add_trace(
                    go.Scatter(
                        x=list(range(252)),
                        y=avg_path,
                        mode='lines',
                        name='Average Path',
                        line=dict(width=3, color='red')
                    )
                )
                
                fig_mc.update_layout(
                    title="Monte Carlo Price Simulation (1 Year)",
                    xaxis_title="Days",
                    yaxis_title="Price Multiple",
                    height=400
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
                
            elif analysis_type == 'Portfolio':
                # Portfolio risk analysis
                st.subheader("💼 Portfolio Risk Analysis")
                
                # Calculate portfolio returns
                portfolio_returns = pd.Series(0, index=asset_data[list(asset_data.keys())[0]].index)
                
                for symbol, weight in portfolio_weights.items():
                    if symbol in asset_data:
                        asset_returns = asset_data[symbol]['Close'].pct_change().dropna()
                        aligned_returns = asset_returns.reindex(portfolio_returns.index, fill_value=0)
                        portfolio_returns += weight * aligned_returns
                
                portfolio_returns = portfolio_returns.dropna()
                portfolio_prices = (1 + portfolio_returns).cumprod()
                
                # Portfolio risk metrics
                portfolio_risk = risk_calculator.risk_metrics_summary(
                    portfolio_prices,
                    benchmark_prices=benchmark_data['Close'] if not benchmark_data.empty else None
                )
                
                # Display portfolio metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Portfolio Return", f"{portfolio_risk['annual_return']:.1%}")
                    st.metric("Portfolio Volatility", f"{portfolio_risk['annual_volatility']:.1%}")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{portfolio_risk['sharpe_ratio']:.3f}")
                    st.metric("Sortino Ratio", f"{portfolio_risk['sortino_ratio']:.3f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{portfolio_risk['max_drawdown']:.1%}")
                    st.metric("VaR (95%)", f"{portfolio_risk['var_95']:.2%}")
                
                with col4:
                    st.metric("CVaR (95%)", f"{portfolio_risk['cvar_95']:.2%}")
                    st.metric("Calmar Ratio", f"{portfolio_risk['calmar_ratio']:.3f}")
                
                # Portfolio composition
                st.subheader("📊 Portfolio Composition")
                
                weights_df = pd.DataFrame(
                    list(portfolio_weights.items()),
                    columns=['Asset', 'Weight']
                )
                weights_df['Weight_Pct'] = weights_df['Weight'] * 100
                
                fig_pie = px.pie(
                    weights_df,
                    values='Weight',
                    names='Asset',
                    title="Portfolio Allocation"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Correlation matrix
                st.subheader("🔗 Asset Correlation Matrix")
                
                returns_matrix = pd.DataFrame()
                for symbol in portfolio_weights.keys():
                    if symbol in asset_data:
                        returns_matrix[symbol] = asset_data[symbol]['Close'].pct_change()
                
                correlation_matrix = returns_matrix.corr()
                
                fig_corr = px.imshow(
                    correlation_matrix,
                    title="Asset Correlation Matrix",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
            else:  # Comparative Analysis
                # Comparative risk analysis
                st.subheader("🔄 Comparative Risk Analysis")
                
                # Risk comparison table
                comparison_data = []
                for symbol in selected_assets:
                    if symbol in asset_data:
                        asset_prices = asset_data[symbol]['Close']
                        asset_returns = asset_prices.pct_change().dropna()
                        
                        risk_metrics_asset = risk_calculator.risk_metrics_summary(asset_prices)
                        
                        comparison_data.append({
                            'Asset': symbol,
                            'Annual Return': f"{risk_metrics_asset['annual_return']:.1%}",
                            'Volatility': f"{risk_metrics_asset['annual_volatility']:.1%}",
                            'Sharpe Ratio': f"{risk_metrics_asset['sharpe_ratio']:.3f}",
                            'Max Drawdown': f"{risk_metrics_asset['max_drawdown']:.1%}",
                            'VaR (95%)': f"{risk_metrics_asset['var_95']:.2%}",
                            'Skewness': f"{risk_metrics_asset['skewness']:.2f}",
                            'Kurtosis': f"{risk_metrics_asset['kurtosis']:.2f}"
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Risk-return scatter plot
                st.subheader("📊 Risk-Return Analysis")
                
                returns_list = []
                volatility_list = []
                sharpe_list = []
                assets_list = []
                
                for symbol in selected_assets:
                    if symbol in asset_data:
                        asset_returns = asset_data[symbol]['Close'].pct_change().dropna()
                        annual_return = asset_returns.mean() * 252
                        annual_vol = asset_returns.std() * np.sqrt(252)
                        sharpe = risk_calculator.sharpe_ratio(asset_returns)
                        
                        returns_list.append(annual_return)
                        volatility_list.append(annual_vol)
                        sharpe_list.append(sharpe)
                        assets_list.append(symbol)
                
                fig_scatter = go.Figure()
                
                fig_scatter.add_trace(
                    go.Scatter(
                        x=volatility_list,
                        y=returns_list,
                        mode='markers+text',
                        text=assets_list,
                        textposition='top center',
                        marker=dict(
                            size=15,
                            color=sharpe_list,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        name='Assets'
                    )
                )
                
                fig_scatter.update_layout(
                    title="Risk-Return Profile",
                    xaxis_title="Volatility (Annual)",
                    yaxis_title="Expected Return (Annual)",
                    height=500
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Stress testing (if enabled)
            if stress_test_enabled:
                st.subheader("🔥 Stress Testing")
                
                # Define stress scenarios
                stress_scenarios = {
                    "Market Crash (-30%)": {"return_shock": -0.30, "volatility_multiplier": 2.0},
                    "High Volatility": {"return_shock": 0.0, "volatility_multiplier": 3.0},
                    "Recession": {"return_shock": -0.20, "volatility_multiplier": 1.5},
                    "Black Swan": {"return_shock": -0.50, "volatility_multiplier": 4.0},
                    "Stagflation": {"return_shock": -0.10, "volatility_multiplier": 1.8}
                }
                
                # Run stress tests for the first asset (or portfolio)
                target_asset = selected_assets[0]
                if target_asset in asset_data:
                    asset_prices = asset_data[target_asset]['Close']
                    
                    stress_results = risk_calculator.stress_testing(asset_prices, stress_scenarios)
                    
                    stress_df = pd.DataFrame(stress_results).T
                    stress_df.columns = ['Price Change', 'VaR (95%)', 'Volatility', 'Sharpe Ratio']
                    
                    # Format the dataframe for display
                    stress_display = stress_df.copy()
                    stress_display['Price Change'] = stress_display['Price Change'].apply(lambda x: f"{x:.1%}")
                    stress_display['VaR (95%)'] = stress_display['VaR (95%)'].apply(lambda x: f"{x:.2%}")
                    stress_display['Volatility'] = stress_display['Volatility'].apply(lambda x: f"{x:.1%}")
                    stress_display['Sharpe Ratio'] = stress_display['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(stress_display, use_container_width=True)
                    
                    # Stress test visualization
                    fig_stress = go.Figure()
                    
                    scenarios = list(stress_scenarios.keys())
                    price_changes = [stress_results[scenario]['price_change'] * 100 for scenario in scenarios]
                    
                    fig_stress.add_trace(
                        go.Bar(
                            x=scenarios,
                            y=price_changes,
                            name='Price Impact',
                            marker_color=['red' if x < 0 else 'green' for x in price_changes]
                        )
                    )
                    
                    fig_stress.update_layout(
                        title="Stress Test Results - Price Impact",
                        xaxis_title="Scenario",
                        yaxis_title="Price Change (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_stress, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error during risk analysis: {str(e)}")
            st.info("Please check your inputs and try again.")

# Educational section
with st.expander("📚 Understanding Risk Assessment"):
    st.markdown("""
    ### Key Risk Metrics Explained
    
    **Value at Risk (VaR):**
    - Maximum expected loss at a given confidence level
    - 95% VaR means 5% chance of losing more than this amount
    
    **Conditional Value at Risk (CVaR):**
    - Average loss beyond the VaR threshold
    - Also known as Expected Shortfall
    
    **Maximum Drawdown:**
    - Largest peak-to-trough decline in portfolio value
    - Measures worst-case scenario
    
    **Sharpe Ratio:**
    - Risk-adjusted return measure
    - Higher values indicate better risk-adjusted performance
    
    **Sortino Ratio:**
    - Similar to Sharpe but only considers downside volatility
    - Better for asymmetric return distributions
    
    **Ulcer Index:**
    - Measures downside volatility and duration of drawdowns
    - Lower values indicate less downside risk
    
    ### Distribution Analysis
    - **Skewness**: Measures asymmetry of returns (negative = more downside risk)
    - **Kurtosis**: Measures tail risk (higher = more extreme events)
    
    ### Stress Testing
    Tests portfolio performance under extreme market conditions:
    - Market crashes
    - High volatility periods
    - Economic recessions
    - Black swan events
    
    ### Important Notes
    - Risk metrics are based on historical data
    - Past volatility may not predict future risk
    - Correlation between assets can change during crises
    - Always consider multiple risk measures together
    """)

# Footer
st.markdown("---")
st.markdown("*Risk assessment is for educational purposes only and should not be considered as financial advice.*")
