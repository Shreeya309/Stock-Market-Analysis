import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class RiskMetrics:
    def __init__(self, confidence_levels=[0.95, 0.99]):
        """
        Initialize Risk Metrics calculator
        
        Args:
            confidence_levels: List of confidence levels for VaR calculations
        """
        self.confidence_levels = confidence_levels
        
    def calculate_returns(self, prices):
        """
        Calculate returns from price series
        
        Args:
            prices: Price series (pandas Series or DataFrame)
        
        Returns:
            Returns series
        """
        return prices.pct_change().dropna()
    
    def value_at_risk(self, returns, confidence_level=0.95, method='historical'):
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Return series
            confidence_level: Confidence level (default 0.95 for 95% VaR)
            method: 'historical', 'parametric', or 'monte_carlo'
        
        Returns:
            VaR value
        """
        if method == 'historical':
            return np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            mean_return = returns.mean()
            std_return = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return mean_return + z_score * std_return
        
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            num_simulations = 10000
            mean_return = returns.mean()
            std_return = returns.std()
            
            simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
            return np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        else:
            raise ValueError("Method must be 'historical', 'parametric', or 'monte_carlo'")
    
    def conditional_var(self, returns, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk (CVaR) or Expected Shortfall
        
        Args:
            returns: Return series
            confidence_level: Confidence level
        
        Returns:
            CVaR value
        """
        var = self.value_at_risk(returns, confidence_level, method='historical')
        return returns[returns <= var].mean()
    
    def maximum_drawdown(self, prices):
        """
        Calculate Maximum Drawdown
        
        Args:
            prices: Price series
        
        Returns:
            Dictionary with max drawdown, start date, end date, recovery date
        """
        if isinstance(prices, pd.Series):
            cumulative = (1 + self.calculate_returns(prices)).cumprod()
        else:
            cumulative = prices / prices.iloc[0]
        
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find start of drawdown period
        start_date = running_max[running_max.index <= max_dd_date].idxmax()
        
        # Find recovery date (if any)
        recovery_date = None
        post_dd = cumulative[cumulative.index > max_dd_date]
        if len(post_dd) > 0:
            recovery_level = running_max.loc[max_dd_date]
            recovery_series = post_dd[post_dd >= recovery_level]
            if len(recovery_series) > 0:
                recovery_date = recovery_series.index[0]
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date,
            'start_date': start_date,
            'recovery_date': recovery_date,
            'drawdown_series': drawdown
        }
    
    def volatility_metrics(self, returns, window=252):
        """
        Calculate various volatility metrics
        
        Args:
            returns: Return series
            window: Rolling window for calculations
        
        Returns:
            Dictionary with volatility metrics
        """
        # Historical volatility (annualized)
        historical_vol = returns.std() * np.sqrt(252)
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=min(window, len(returns))).std() * np.sqrt(252)
        
        # EWMA volatility
        ewma_vol = returns.ewm(span=window).std() * np.sqrt(252)
        
        # GARCH(1,1) parameters estimation (simplified)
        try:
            garch_vol = self._estimate_garch_volatility(returns)
        except:
            garch_vol = historical_vol
        
        return {
            'historical_volatility': historical_vol,
            'rolling_volatility': rolling_vol,
            'ewma_volatility': ewma_vol,
            'garch_volatility': garch_vol
        }
    
    def _estimate_garch_volatility(self, returns, p=1, q=1):
        """
        Simple GARCH(1,1) volatility estimation
        """
        # Initialize parameters
        omega = 0.01
        alpha = 0.1
        beta = 0.8
        
        # Calculate conditional variances
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = returns.var()
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]
        
        # Return annualized volatility
        return np.sqrt(sigma2[-1] * 252)
    
    def beta_calculation(self, stock_returns, market_returns):
        """
        Calculate Beta (systematic risk measure)
        
        Args:
            stock_returns: Stock return series
            market_returns: Market return series
        
        Returns:
            Beta value
        """
        # Align the series
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return np.nan
        
        stock_aligned = aligned_data.iloc[:, 0]
        market_aligned = aligned_data.iloc[:, 1]
        
        covariance = np.cov(stock_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)
        
        return covariance / market_variance if market_variance != 0 else np.nan
    
    def tracking_error(self, portfolio_returns, benchmark_returns):
        """
        Calculate tracking error
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
        
        Returns:
            Tracking error (annualized)
        """
        # Align returns
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return np.nan
        
        excess_returns = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
        return excess_returns.std() * np.sqrt(252)
    
    def information_ratio(self, portfolio_returns, benchmark_returns):
        """
        Calculate Information Ratio
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
        
        Returns:
            Information ratio
        """
        # Align returns
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return np.nan
        
        excess_returns = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
        tracking_err = self.tracking_error(portfolio_returns, benchmark_returns)
        
        return excess_returns.mean() * 252 / tracking_err if tracking_err != 0 else np.nan
    
    def sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate Sharpe Ratio
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Sharpe ratio
        """
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        
        return excess_returns / volatility if volatility != 0 else np.nan
    
    def sortino_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate Sortino Ratio (uses downside deviation)
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Sortino ratio
        """
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        return excess_returns / downside_deviation if downside_deviation != 0 else np.nan
    
    def calmar_ratio(self, prices, risk_free_rate=0.02):
        """
        Calculate Calmar Ratio
        
        Args:
            prices: Price series
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Calmar ratio
        """
        returns = self.calculate_returns(prices)
        annual_return = returns.mean() * 252 - risk_free_rate
        max_dd = abs(self.maximum_drawdown(prices)['max_drawdown'])
        
        return annual_return / max_dd if max_dd != 0 else np.nan
    
    def ulcer_index(self, prices):
        """
        Calculate Ulcer Index (downside risk measure)
        
        Args:
            prices: Price series
        
        Returns:
            Ulcer Index
        """
        drawdown_series = self.maximum_drawdown(prices)['drawdown_series']
        ulcer = np.sqrt((drawdown_series ** 2).mean())
        
        return ulcer
    
    def risk_metrics_summary(self, prices, benchmark_prices=None, risk_free_rate=0.02):
        """
        Calculate comprehensive risk metrics summary
        
        Args:
            prices: Price series
            benchmark_prices: Benchmark price series (optional)
            risk_free_rate: Risk-free rate
        
        Returns:
            Dictionary with all risk metrics
        """
        returns = self.calculate_returns(prices)
        
        # Basic metrics
        metrics = {
            'total_return': (prices.iloc[-1] / prices.iloc[0] - 1),
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': self.sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': self.sortino_ratio(returns, risk_free_rate),
            'calmar_ratio': self.calmar_ratio(prices, risk_free_rate)
        }
        
        # Drawdown metrics
        dd_metrics = self.maximum_drawdown(prices)
        metrics.update({
            'max_drawdown': dd_metrics['max_drawdown'],
            'max_drawdown_date': dd_metrics['max_drawdown_date'],
            'ulcer_index': self.ulcer_index(prices)
        })
        
        # VaR metrics
        for cl in self.confidence_levels:
            metrics[f'var_{int(cl*100)}'] = self.value_at_risk(returns, cl)
            metrics[f'cvar_{int(cl*100)}'] = self.conditional_var(returns, cl)
        
        # Volatility metrics
        vol_metrics = self.volatility_metrics(returns)
        metrics.update(vol_metrics)
        
        # Benchmark-relative metrics
        if benchmark_prices is not None:
            benchmark_returns = self.calculate_returns(benchmark_prices)
            
            metrics.update({
                'beta': self.beta_calculation(returns, benchmark_returns),
                'tracking_error': self.tracking_error(returns, benchmark_returns),
                'information_ratio': self.information_ratio(returns, benchmark_returns)
            })
        
        # Additional risk measures
        metrics.update({
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'win_rate': (returns > 0).mean(),
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else np.nan
        })
        
        return metrics
    
    def monte_carlo_simulation(self, prices, num_simulations=10000, days=252):
        """
        Monte Carlo simulation for risk assessment
        
        Args:
            prices: Price series
            num_simulations: Number of simulations
            days: Number of days to simulate
        
        Returns:
            Dictionary with simulation results
        """
        returns = self.calculate_returns(prices)
        
        # Estimate parameters
        mu = returns.mean()
        sigma = returns.std()
        
        # Simulate paths
        dt = 1
        simulated_returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), (days, num_simulations))
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + simulated_returns, axis=0)
        final_values = cumulative_returns[-1]
        
        # Calculate risk metrics
        var_95 = np.percentile(final_values, 5)
        var_99 = np.percentile(final_values, 1)
        expected_return = np.mean(final_values)
        
        return {
            'simulated_paths': cumulative_returns,
            'final_values': final_values,
            'expected_final_value': expected_return,
            'var_95': var_95 - 1,  # Convert to loss
            'var_99': var_99 - 1,
            'probability_of_loss': np.mean(final_values < 1)
        }
    
    def stress_testing(self, prices, scenarios):
        """
        Stress testing with different market scenarios
        
        Args:
            prices: Price series
            scenarios: Dictionary of scenarios with return adjustments
        
        Returns:
            Dictionary with stress test results
        """
        returns = self.calculate_returns(prices)
        current_price = prices.iloc[-1]
        
        stress_results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            # Apply scenario stress
            if 'return_shock' in scenario_params:
                shocked_return = returns.mean() + scenario_params['return_shock']
            else:
                shocked_return = returns.mean()
            
            if 'volatility_multiplier' in scenario_params:
                shocked_vol = returns.std() * scenario_params['volatility_multiplier']
            else:
                shocked_vol = returns.std()
            
            # Calculate stressed price
            stressed_price = current_price * (1 + shocked_return)
            
            # Calculate new risk metrics
            stressed_returns = np.random.normal(shocked_return, shocked_vol, len(returns))
            stressed_returns = pd.Series(stressed_returns)
            
            stress_results[scenario_name] = {
                'price_change': (stressed_price - current_price) / current_price,
                'var_95': self.value_at_risk(stressed_returns, 0.95),
                'volatility': shocked_vol * np.sqrt(252),
                'sharpe_ratio': self.sharpe_ratio(stressed_returns)
            }
        
        return stress_results
