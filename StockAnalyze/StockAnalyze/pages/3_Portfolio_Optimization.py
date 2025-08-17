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

from models.portfolio_optimizer import PortfolioOptimizer
from utils.data_fetcher import StockDataFetcher
from utils.risk_metrics import RiskMetrics

st.set_page_config(page_title="Portfolio Optimization", page_icon="💼", layout="wide")

st.title("💼 Modern Portfolio Theory Optimization")

# Initialize components
data_fetcher = StockDataFetcher()
risk_metrics = RiskMetrics()

# Sidebar controls
st.sidebar.header("⚖️ Portfolio Settings")

# Portfolio configuration
st.sidebar.subheader("📊 Asset Selection")

# Predefined portfolios
portfolio_presets = {
    "Tech Giants": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    "FAANG": ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL'],
    "Blue Chip": ['AAPL', 'MSFT', 'JNJ', 'JPM', 'PG'],
    "Growth Stocks": ['TSLA', 'NVDA', 'CRM', 'ZOOM', 'SQ'],
    "Dividend Aristocrats": ['KO', 'PG', 'JNJ', 'MCD', 'WMT'],
    "Custom": []
}

preset_choice = st.sidebar.selectbox(
    "Select Portfolio Preset",
    list(portfolio_presets.keys()),
    help="Choose a predefined portfolio or create custom"
)

if preset_choice == "Custom":
    # Custom portfolio input
    st.sidebar.write("Enter stock symbols (one per line):")
    custom_symbols = st.sidebar.text_area(
        "Stock Symbols",
        placeholder="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA",
        height=100
    )
    
    if custom_symbols:
        selected_symbols = [symbol.strip().upper() for symbol in custom_symbols.split('\n') if symbol.strip()]
    else:
        selected_symbols = ['AAPL', 'MSFT', 'GOOGL']
else:
    selected_symbols = portfolio_presets[preset_choice]

# Optimization parameters
st.sidebar.subheader("🎯 Optimization Parameters")

optimization_method = st.sidebar.selectbox(
    "Optimization Method",
    ['Maximum Sharpe Ratio', 'Minimum Variance', 'Risk Parity', 'Target Return'],
    help="Choose optimization objective"
)

if optimization_method == 'Target Return':
    target_return = st.sidebar.slider(
        "Target Annual Return (%)",
        min_value=1.0,
        max_value=30.0,
        value=12.0,
        step=0.5
    ) / 100

data_period = st.sidebar.selectbox(
    "Historical Data Period",
    ['1y', '2y', '3y', '5y'],
    index=1,
    help="Period for historical return calculation"
)

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1
) / 100

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    include_constraints = st.checkbox("Include Position Constraints")
    if include_constraints:
        max_weight = st.slider("Maximum Weight per Asset (%)", 10, 50, 30) / 100
        min_weight = st.slider("Minimum Weight per Asset (%)", 0, 10, 5) / 100
    
    enable_rebalancing = st.checkbox("Enable Rebalancing Analysis")
    if enable_rebalancing:
        rebalance_frequency = st.selectbox("Rebalancing Frequency", ['Monthly', 'Quarterly', 'Semi-Annual', 'Annual'])
    
    monte_carlo_sims = st.slider("Monte Carlo Simulations", 1000, 50000, 10000, step=1000)

# Main content
if st.button("🚀 Optimize Portfolio", type="primary"):
    
    if len(selected_symbols) < 2:
        st.error("Please select at least 2 stocks for portfolio optimization.")
        st.stop()
    
    with st.spinner("Optimizing portfolio..."):
        try:
            # Initialize portfolio optimizer
            optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
            
            # Fetch data for all symbols
            st.info(f"Fetching data for {len(selected_symbols)} assets...")
            portfolio_data = optimizer.fetch_data(selected_symbols, period=data_period)
            
            if portfolio_data.empty:
                st.error("Failed to fetch data for the selected symbols.")
                st.stop()
            
            # Get individual stock info
            stock_infos = {}
            for symbol in selected_symbols:
                try:
                    stock_infos[symbol] = data_fetcher.get_stock_info(symbol)
                except:
                    stock_infos[symbol] = {'company_name': symbol, 'market_cap': 0}
            
            # Display portfolio overview
            st.subheader("📊 Portfolio Overview")
            
            # Create overview table
            overview_data = []
            for symbol in selected_symbols:
                current_price = portfolio_data[symbol].iloc[-1]
                returns = portfolio_data[symbol].pct_change().dropna()
                annual_return = returns.mean() * 252
                annual_vol = returns.std() * np.sqrt(252)
                
                overview_data.append({
                    'Symbol': symbol,
                    'Company': stock_infos[symbol].get('company_name', symbol)[:30],
                    'Current Price': f"${current_price:.2f}",
                    'Annual Return': f"{annual_return:.1%}",
                    'Annual Volatility': f"{annual_vol:.1%}",
                    'Market Cap': f"${stock_infos[symbol].get('market_cap', 0) / 1e9:.1f}B" if stock_infos[symbol].get('market_cap', 0) > 0 else "N/A"
                })
            
            overview_df = pd.DataFrame(overview_data)
            st.dataframe(overview_df, use_container_width=True)
            
            # Correlation matrix
            st.subheader("🔗 Asset Correlation Matrix")
            
            returns_data = portfolio_data.pct_change().dropna()
            correlation_matrix = returns_data.corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                title="Asset Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Perform optimization
            st.subheader("⚖️ Portfolio Optimization Results")
            
            optimization_results = {}
            
            if optimization_method == 'Maximum Sharpe Ratio':
                result = optimizer.optimize_sharpe()
                optimization_results['Optimal'] = result
                
            elif optimization_method == 'Minimum Variance':
                result = optimizer.optimize_minimum_variance()
                optimization_results['Optimal'] = result
                
            elif optimization_method == 'Risk Parity':
                result = optimizer.risk_parity_optimization()
                optimization_results['Optimal'] = result
                
            elif optimization_method == 'Target Return':
                result = optimizer.optimize_target_return(target_return)
                optimization_results['Optimal'] = result
            
            # Also calculate other portfolios for comparison
            try:
                optimization_results['Max Sharpe'] = optimizer.optimize_sharpe()
                optimization_results['Min Variance'] = optimizer.optimize_minimum_variance()
                optimization_results['Risk Parity'] = optimizer.risk_parity_optimization()
            except:
                pass
            
            # Display optimization results
            if optimization_results:
                optimal_result = optimization_results['Optimal']
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Expected Return",
                        f"{optimal_result['expected_return']:.1%}"
                    )
                
                with col2:
                    st.metric(
                        "Volatility",
                        f"{optimal_result['volatility']:.1%}"
                    )
                
                with col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{optimal_result['sharpe_ratio']:.3f}"
                    )
                
                with col4:
                    optimization_type = optimal_result.get('method', optimization_method)
                    st.metric(
                        "Optimization Type",
                        optimization_type
                    )
                
                # Portfolio weights
                st.subheader("📈 Optimal Portfolio Weights")
                
                weights_df = pd.DataFrame(
                    list(optimal_result['weights'].items()),
                    columns=['Asset', 'Weight']
                )
                weights_df['Weight_Pct'] = weights_df['Weight'] * 100
                weights_df['Company'] = weights_df['Asset'].map(
                    lambda x: stock_infos[x].get('company_name', x)[:30]
                )
                
                # Pie chart
                fig_pie = px.pie(
                    weights_df,
                    values='Weight',
                    names='Asset',
                    title="Portfolio Allocation",
                    hover_data=['Company']
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Weights table
                weights_display = weights_df[['Asset', 'Company', 'Weight_Pct']].copy()
                weights_display['Weight_Pct'] = weights_display['Weight_Pct'].apply(lambda x: f"{x:.1f}%")
                weights_display.columns = ['Symbol', 'Company', 'Weight (%)']
                st.dataframe(weights_display, use_container_width=True)
                
                # Efficient frontier
                st.subheader("📊 Efficient Frontier")
                
                with st.spinner("Calculating efficient frontier..."):
                    try:
                        efficient_portfolios = optimizer.efficient_frontier(num_portfolios=50)
                        
                        if efficient_portfolios:
                            # Extract returns and volatilities
                            frontier_returns = [p['expected_return'] for p in efficient_portfolios]
                            frontier_vols = [p['volatility'] for p in efficient_portfolios]
                            frontier_sharpes = [p['sharpe_ratio'] for p in efficient_portfolios]
                            
                            # Create efficient frontier plot
                            fig_frontier = go.Figure()
                            
                            # Efficient frontier
                            fig_frontier.add_trace(
                                go.Scatter(
                                    x=frontier_vols,
                                    y=frontier_returns,
                                    mode='lines+markers',
                                    name='Efficient Frontier',
                                    marker=dict(
                                        color=frontier_sharpes,
                                        colorscale='Viridis',
                                        showscale=True,
                                        colorbar=dict(title="Sharpe Ratio")
                                    )
                                )
                            )
                            
                            # Add optimization results
                            for name, result in optimization_results.items():
                                fig_frontier.add_trace(
                                    go.Scatter(
                                        x=[result['volatility']],
                                        y=[result['expected_return']],
                                        mode='markers',
                                        name=name,
                                        marker=dict(size=12, symbol='star')
                                    )
                                )
                            
                            # Individual assets
                            individual_returns = []
                            individual_vols = []
                            for symbol in selected_symbols:
                                returns = returns_data[symbol]
                                individual_returns.append(returns.mean() * 252)
                                individual_vols.append(returns.std() * np.sqrt(252))
                            
                            fig_frontier.add_trace(
                                go.Scatter(
                                    x=individual_vols,
                                    y=individual_returns,
                                    mode='markers+text',
                                    name='Individual Assets',
                                    text=selected_symbols,
                                    textposition='top center',
                                    marker=dict(size=8, color='red')
                                )
                            )
                            
                            fig_frontier.update_layout(
                                title="Efficient Frontier Analysis",
                                xaxis_title="Volatility (Annual)",
                                yaxis_title="Expected Return (Annual)",
                                height=500
                            )
                            
                            st.plotly_chart(fig_frontier, use_container_width=True)
                    
                    except Exception as e:
                        st.warning(f"Could not calculate efficient frontier: {str(e)}")
                
                # Monte Carlo simulation
                st.subheader("🎲 Monte Carlo Simulation")
                
                with st.spinner("Running Monte Carlo simulation..."):
                    try:
                        optimal_weights = np.array([optimal_result['weights'][symbol] for symbol in selected_symbols])
                        mc_results = optimizer.monte_carlo_simulation(
                            optimal_weights,
                            num_simulations=monte_carlo_sims,
                            time_horizon=252
                        )
                        
                        # Monte Carlo results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Expected Return (1Y)",
                                f"{(mc_results['expected_final_value'] - 1):.1%}"
                            )
                        
                        with col2:
                            st.metric(
                                "95% VaR (1Y)",
                                f"{mc_results['var_95']:.1%}"
                            )
                        
                        with col3:
                            st.metric(
                                "Probability of Loss",
                                f"{mc_results['probability_of_loss']:.1%}"
                            )
                        
                        # Monte Carlo distribution
                        fig_mc = go.Figure()
                        
                        final_returns = (mc_results['final_values'] - 1) * 100
                        
                        fig_mc.add_trace(
                            go.Histogram(
                                x=final_returns,
                                nbinsx=50,
                                name='Return Distribution',
                                opacity=0.7
                            )
                        )
                        
                        # Add VaR lines
                        var_95 = mc_results['var_95'] * 100
                        var_99 = mc_results['var_99'] * 100
                        
                        fig_mc.add_vline(
                            x=var_95,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"95% VaR: {var_95:.1f}%"
                        )
                        
                        fig_mc.add_vline(
                            x=var_99,
                            line_dash="dash",
                            line_color="darkred",
                            annotation_text=f"99% VaR: {var_99:.1f}%"
                        )
                        
                        fig_mc.update_layout(
                            title="Monte Carlo Return Distribution (1 Year)",
                            xaxis_title="Annual Return (%)",
                            yaxis_title="Frequency",
                            height=400
                        )
                        
                        st.plotly_chart(fig_mc, use_container_width=True)
                        
                        # Sample paths
                        st.subheader("📈 Sample Portfolio Paths")
                        
                        fig_paths = go.Figure()
                        
                        # Show subset of paths
                        num_paths_to_show = min(100, mc_results['simulated_paths'].shape[1])
                        sample_indices = np.random.choice(
                            mc_results['simulated_paths'].shape[1],
                            num_paths_to_show,
                            replace=False
                        )
                        
                        for i in sample_indices:
                            fig_paths.add_trace(
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
                        
                        # Add average path
                        avg_path = np.mean(mc_results['simulated_paths'], axis=1)
                        fig_paths.add_trace(
                            go.Scatter(
                                x=list(range(252)),
                                y=avg_path,
                                mode='lines',
                                name='Average Path',
                                line=dict(width=3, color='red')
                            )
                        )
                        
                        fig_paths.update_layout(
                            title="Monte Carlo Portfolio Simulation Paths",
                            xaxis_title="Days",
                            yaxis_title="Portfolio Value",
                            height=400
                        )
                        
                        st.plotly_chart(fig_paths, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Could not run Monte Carlo simulation: {str(e)}")
                
                # Backtesting
                st.subheader("📊 Portfolio Backtesting")
                
                with st.spinner("Running backtest..."):
                    try:
                        # Split data for backtesting
                        split_date = portfolio_data.index[int(len(portfolio_data) * 0.7)]
                        
                        backtest_results = optimizer.backtest_strategy(
                            optimal_result['weights'],
                            start_date=split_date
                        )
                        
                        # Backtest metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Total Return",
                                f"{backtest_results['total_return']:.1%}"
                            )
                        
                        with col2:
                            st.metric(
                                "Annual Return",
                                f"{backtest_results['annualized_return']:.1%}"
                            )
                        
                        with col3:
                            st.metric(
                                "Sharpe Ratio",
                                f"{backtest_results['sharpe_ratio']:.3f}"
                            )
                        
                        with col4:
                            st.metric(
                                "Max Drawdown",
                                f"{backtest_results['max_drawdown']:.1%}"
                            )
                        
                        # Backtest chart
                        fig_backtest = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=['Cumulative Returns', 'Drawdown'],
                            vertical_spacing=0.1,
                            row_heights=[0.7, 0.3]
                        )
                        
                        # Cumulative returns
                        fig_backtest.add_trace(
                            go.Scatter(
                                x=backtest_results['cumulative_returns'].index,
                                y=backtest_results['cumulative_returns'],
                                mode='lines',
                                name='Portfolio',
                                line=dict(color='blue', width=2)
                            ),
                            row=1, col=1
                        )
                        
                        # Drawdown
                        fig_backtest.add_trace(
                            go.Scatter(
                                x=backtest_results['drawdown'].index,
                                y=backtest_results['drawdown'],
                                mode='lines',
                                name='Drawdown',
                                fill='tonexty',
                                line=dict(color='red'),
                                fillcolor='rgba(255,0,0,0.3)'
                            ),
                            row=2, col=1
                        )
                        
                        fig_backtest.update_layout(
                            title="Portfolio Backtest Results",
                            height=500,
                            showlegend=True
                        )
                        
                        fig_backtest.update_xaxes(title_text="Date", row=2, col=1)
                        fig_backtest.update_yaxes(title_text="Cumulative Return", row=1, col=1)
                        fig_backtest.update_yaxes(title_text="Drawdown", row=2, col=1)
                        
                        st.plotly_chart(fig_backtest, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Could not run backtest: {str(e)}")
                
                # Risk analysis
                st.subheader("⚠️ Risk Analysis")
                
                portfolio_returns = (portfolio_data * optimal_weights).sum(axis=1).pct_change().dropna()
                risk_summary = risk_metrics.risk_metrics_summary(
                    (1 + portfolio_returns).cumprod(),
                    risk_free_rate=risk_free_rate
                )
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Value at Risk (95%)", f"{risk_summary['var_95']:.2%}")
                    st.metric("Conditional VaR (95%)", f"{risk_summary['cvar_95']:.2%}")
                
                with col2:
                    st.metric("Maximum Drawdown", f"{risk_summary['max_drawdown']:.2%}")
                    st.metric("Ulcer Index", f"{risk_summary['ulcer_index']:.4f}")
                
                with col3:
                    st.metric("Skewness", f"{risk_summary['skewness']:.3f}")
                    st.metric("Kurtosis", f"{risk_summary['kurtosis']:.3f}")
                
                with col4:
                    st.metric("Win Rate", f"{risk_summary['win_rate']:.1%}")
                    st.metric("Profit Factor", f"{risk_summary.get('profit_factor', 0):.2f}")
                
            else:
                st.error("Optimization failed. Please try with different parameters.")
                
        except Exception as e:
            st.error(f"Error during portfolio optimization: {str(e)}")
            st.info("Please check your stock symbols and try again.")

# Educational section
with st.expander("📚 Understanding Portfolio Optimization"):
    st.markdown("""
    ### Modern Portfolio Theory (MPT)
    
    **Core Concepts:**
    - **Efficient Frontier**: The set of optimal portfolios offering the highest expected return for each level of risk
    - **Sharpe Ratio**: Risk-adjusted return measure (higher is better)
    - **Diversification**: Reducing risk through asset allocation
    
    **Optimization Methods:**
    
    **Maximum Sharpe Ratio:**
    - Maximizes return per unit of risk
    - Best risk-adjusted performance
    
    **Minimum Variance:**
    - Minimizes portfolio volatility
    - Conservative approach for risk-averse investors
    
    **Risk Parity:**
    - Equal risk contribution from each asset
    - Alternative to market-cap weighting
    
    **Target Return:**
    - Achieves specific return target with minimum risk
    - Goal-based optimization
    
    ### Key Metrics
    - **Expected Return**: Projected annual return based on historical data
    - **Volatility**: Standard deviation of returns (risk measure)
    - **Value at Risk (VaR)**: Maximum expected loss at given confidence level
    - **Maximum Drawdown**: Largest peak-to-trough decline
    
    ### Important Considerations
    - Based on historical data - future performance may differ
    - Assumes normal distribution of returns
    - Correlation between assets can change over time
    - Should be combined with fundamental analysis
    """)

# Footer
st.markdown("---")
st.markdown("*Portfolio optimization results are for educational purposes only and should not be considered as investment advice.*")
