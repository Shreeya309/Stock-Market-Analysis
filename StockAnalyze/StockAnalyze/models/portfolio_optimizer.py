import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize Portfolio Optimizer
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.symbols = None
        
    def fetch_data(self, symbols, period='2y'):
        """
        Fetch historical data for portfolio optimization
        """
        self.symbols = symbols
        
        # Download data
        data = yf.download(symbols, period=period)['Adj Close']
        
        if len(symbols) == 1:
            data = data.to_frame()
            data.columns = symbols
        
        # Calculate returns
        self.returns = data.pct_change().dropna()
        self.mean_returns = self.returns.mean() * 252  # Annualized
        self.cov_matrix = self.returns.cov() * 252     # Annualized
        
        return data
    
    def portfolio_stats(self, weights):
        """
        Calculate portfolio statistics
        """
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return portfolio_return, portfolio_vol, sharpe_ratio
    
    def negative_sharpe(self, weights):
        """
        Negative Sharpe ratio for minimization
        """
        return -self.portfolio_stats(weights)[2]
    
    def portfolio_volatility(self, weights):
        """
        Calculate portfolio volatility
        """
        return self.portfolio_stats(weights)[1]
    
    def optimize_sharpe(self):
        """
        Optimize portfolio for maximum Sharpe ratio
        """
        num_assets = len(self.symbols)
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            self.negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_stats(optimal_weights)
            
            return {
                'weights': dict(zip(self.symbols, optimal_weights)),
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'optimization_result': result
            }
        else:
            raise ValueError("Optimization failed")
    
    def optimize_minimum_variance(self):
        """
        Optimize portfolio for minimum variance
        """
        num_assets = len(self.symbols)
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            self.portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_stats(optimal_weights)
            
            return {
                'weights': dict(zip(self.symbols, optimal_weights)),
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'optimization_result': result
            }
        else:
            raise ValueError("Optimization failed")
    
    def efficient_frontier(self, num_portfolios=100):
        """
        Generate efficient frontier
        """
        if self.returns is None:
            raise ValueError("No data available. Call fetch_data first.")
        
        # Get min variance portfolio
        min_var_portfolio = self.optimize_minimum_variance()
        min_vol = min_var_portfolio['volatility']
        
        # Get max return
        max_ret = self.mean_returns.max()
        
        # Create target returns
        target_returns = np.linspace(min_var_portfolio['expected_return'], max_ret, num_portfolios)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            try:
                portfolio = self.optimize_target_return(target_ret)
                efficient_portfolios.append(portfolio)
            except:
                continue
        
        return efficient_portfolios
    
    def optimize_target_return(self, target_return):
        """
        Optimize portfolio for a target return
        """
        num_assets = len(self.symbols)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(x * self.mean_returns) - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            self.portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_stats(optimal_weights)
            
            return {
                'weights': dict(zip(self.symbols, optimal_weights)),
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'target_return': target_return
            }
        else:
            raise ValueError(f"Optimization failed for target return {target_return}")
    
    def black_litterman_optimization(self, market_caps, tau=0.025, views=None, omega=None):
        """
        Black-Litterman portfolio optimization
        
        Args:
            market_caps: Market capitalizations for each asset
            tau: Scaling parameter
            views: Investor views (P matrix)
            omega: Uncertainty matrix for views
        """
        n = len(self.symbols)
        
        # Market capitalization weights
        w_market = np.array(list(market_caps.values()))
        w_market = w_market / w_market.sum()
        
        # Implied equilibrium returns
        delta = (self.mean_returns.mean() - self.risk_free_rate) / (w_market @ self.cov_matrix @ w_market)
        pi = delta * (self.cov_matrix @ w_market)
        
        if views is None or omega is None:
            # No views, return market portfolio
            ret, vol, sharpe = self.portfolio_stats(w_market)
            return {
                'weights': dict(zip(self.symbols, w_market)),
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'method': 'market_portfolio'
            }
        
        # Black-Litterman calculation
        tau_cov = tau * self.cov_matrix
        M1 = np.linalg.inv(tau_cov)
        M2 = views.T @ np.linalg.inv(omega) @ views
        M3 = np.linalg.inv(tau_cov) @ pi
        M4 = views.T @ np.linalg.inv(omega) @ views @ self.mean_returns
        
        # New expected returns
        mu_bl = np.linalg.inv(M1 + M2) @ (M3 + M4)
        
        # New covariance matrix
        cov_bl = np.linalg.inv(M1 + M2)
        
        # Optimize with Black-Litterman inputs
        def bl_objective(weights):
            return -((weights @ mu_bl - self.risk_free_rate) / 
                    np.sqrt(weights @ cov_bl @ weights))
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n))
        initial_weights = w_market
        
        result = minimize(bl_objective, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            ret = optimal_weights @ mu_bl
            vol = np.sqrt(optimal_weights @ cov_bl @ optimal_weights)
            sharpe = (ret - self.risk_free_rate) / vol
            
            return {
                'weights': dict(zip(self.symbols, optimal_weights)),
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'method': 'black_litterman'
            }
        else:
            raise ValueError("Black-Litterman optimization failed")
    
    def risk_parity_optimization(self):
        """
        Risk parity portfolio optimization
        """
        num_assets = len(self.symbols)
        
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = np.ones(num_assets) / num_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.01, 1) for _ in range(num_assets))  # Minimum 1% allocation
        initial_weights = np.array([1/num_assets] * num_assets)
        
        result = minimize(
            risk_budget_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_stats(optimal_weights)
            
            return {
                'weights': dict(zip(self.symbols, optimal_weights)),
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'method': 'risk_parity'
            }
        else:
            raise ValueError("Risk parity optimization failed")
    
    def monte_carlo_simulation(self, weights, num_simulations=10000, time_horizon=252):
        """
        Monte Carlo simulation for portfolio returns
        """
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        
        # Simulate portfolio returns
        simulated_returns = np.random.normal(
            portfolio_return / 252,  # Daily return
            portfolio_vol / np.sqrt(252),  # Daily volatility
            (time_horizon, num_simulations)
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + simulated_returns, axis=0)
        
        # Final portfolio values (assuming initial value of 1)
        final_values = cumulative_returns[-1]
        
        # Calculate statistics
        var_95 = np.percentile(final_values, 5)
        var_99 = np.percentile(final_values, 1)
        expected_return = np.mean(final_values)
        
        return {
            'simulated_paths': cumulative_returns,
            'final_values': final_values,
            'expected_final_value': expected_return,
            'var_95': var_95,
            'var_99': var_99,
            'probability_of_loss': np.mean(final_values < 1)
        }
    
    def backtest_strategy(self, weights, start_date=None, end_date=None):
        """
        Backtest a portfolio strategy
        """
        if start_date is None:
            start_date = self.returns.index[0]
        if end_date is None:
            end_date = self.returns.index[-1]
        
        # Filter returns for backtest period
        backtest_returns = self.returns.loc[start_date:end_date]
        
        # Calculate portfolio returns
        weights_array = np.array([weights[symbol] for symbol in self.symbols])
        portfolio_returns = (backtest_returns * weights_array).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annualized_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_vol
        
        # Maximum drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns,
            'portfolio_returns': portfolio_returns,
            'drawdown': drawdown
        }
