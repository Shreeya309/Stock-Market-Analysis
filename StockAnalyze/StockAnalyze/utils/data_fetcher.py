import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import warnings
warnings.filterwarnings('ignore')

class StockDataFetcher:
    def __init__(self, max_retries=3, retry_delay=1):
        """
        Initialize Stock Data Fetcher
        
        Args:
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def fetch_stock_data(self, symbol, period='2y', interval='1d'):
        """
        Fetch stock data with retry mechanism
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pandas.DataFrame: Stock price data
        """
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if data.empty:
                    raise ValueError(f"No data found for symbol {symbol}")
                
                # Clean the data
                data = data.dropna()
                
                return data
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise Exception(f"Failed to fetch data for {symbol} after {self.max_retries} attempts: {str(e)}")
    
    def fetch_multiple_stocks(self, symbols, period='2y', interval='1d'):
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
        
        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        stock_data = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                data = self.fetch_stock_data(symbol, period, interval)
                stock_data[symbol] = data
                print(f"✓ Successfully fetched data for {symbol}")
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"✗ Failed to fetch data for {symbol}: {str(e)}")
        
        if failed_symbols:
            print(f"Failed to fetch data for: {failed_symbols}")
        
        return stock_data
    
    def get_stock_info(self, symbol):
        """
        Get detailed stock information
        
        Args:
            symbol: Stock symbol
        
        Returns:
            dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            stock_info = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'previous_close': info.get('previousClose', 0),
                'current_price': info.get('currentPrice', 0),
                'target_mean_price': info.get('targetMeanPrice', 0),
                'recommendation': info.get('recommendationMean', 0)
            }
            
            return stock_info
            
        except Exception as e:
            raise Exception(f"Failed to fetch info for {symbol}: {str(e)}")
    
    def get_financial_data(self, symbol):
        """
        Get financial statements data
        
        Args:
            symbol: Stock symbol
        
        Returns:
            dict: Financial data including income statement, balance sheet, cash flow
        """
        try:
            ticker = yf.Ticker(symbol)
            
            financial_data = {
                'income_statement': ticker.financials,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow,
                'quarterly_financials': ticker.quarterly_financials,
                'quarterly_balance_sheet': ticker.quarterly_balance_sheet,
                'quarterly_cashflow': ticker.quarterly_cashflow
            }
            
            return financial_data
            
        except Exception as e:
            print(f"Warning: Could not fetch financial data for {symbol}: {str(e)}")
            return None
    
    def get_market_indices(self):
        """
        Get major market indices data
        
        Returns:
            dict: Market indices data
        """
        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Dow Jones': '^DJI',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }
        
        indices_data = {}
        
        for name, symbol in indices.items():
            try:
                data = self.fetch_stock_data(symbol, period='1y')
                indices_data[name] = data
            except Exception as e:
                print(f"Failed to fetch {name} data: {str(e)}")
        
        return indices_data
    
    def get_sector_performance(self):
        """
        Get sector ETF performance data
        
        Returns:
            dict: Sector performance data
        """
        sectors = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Communication Services': 'XLC',
            'Industrials': 'XLI',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }
        
        sector_data = {}
        
        for sector, etf in sectors.items():
            try:
                data = self.fetch_stock_data(etf, period='1y')
                if not data.empty:
                    # Calculate performance metrics
                    current_price = data['Close'].iloc[-1]
                    start_price = data['Close'].iloc[0]
                    ytd_return = (current_price - start_price) / start_price * 100
                    
                    sector_data[sector] = {
                        'symbol': etf,
                        'current_price': current_price,
                        'ytd_return': ytd_return,
                        'data': data
                    }
                    
            except Exception as e:
                print(f"Failed to fetch {sector} sector data: {str(e)}")
        
        return sector_data
    
    def validate_symbol(self, symbol):
        """
        Validate if a stock symbol exists
        
        Args:
            symbol: Stock symbol to validate
        
        Returns:
            bool: True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got valid data
            if 'symbol' in info or 'shortName' in info or 'longName' in info:
                return True
            else:
                return False
                
        except:
            return False
    
    def get_popular_stocks(self):
        """
        Get a list of popular stocks with their basic info
        
        Returns:
            dict: Popular stocks data
        """
        popular_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'SPOT', 'ZOOM', 'SQ',
            'JPM', 'BAC', 'WFC', 'GS', 'MS',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
            'KO', 'PEP', 'MCD', 'NKE', 'DIS'
        ]
        
        popular_stocks = {}
        
        for symbol in popular_symbols:
            try:
                info = self.get_stock_info(symbol)
                popular_stocks[symbol] = info
            except Exception as e:
                print(f"Warning: Could not fetch info for {symbol}: {str(e)}")
        
        return popular_stocks
    
    def get_real_time_price(self, symbol):
        """
        Get real-time price data
        
        Args:
            symbol: Stock symbol
        
        Returns:
            dict: Real-time price information
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get the most recent data
            data = ticker.history(period="2d", interval="1m")
            
            if data.empty:
                raise ValueError(f"No real-time data available for {symbol}")
            
            current_price = data['Close'].iloc[-1]
            previous_close = ticker.info.get('previousClose', data['Close'].iloc[-2])
            
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'previous_close': previous_close,
                'change': change,
                'change_percent': change_percent,
                'volume': data['Volume'].iloc[-1],
                'timestamp': data.index[-1]
            }
            
        except Exception as e:
            raise Exception(f"Failed to get real-time data for {symbol}: {str(e)}")
    
    def get_earnings_calendar(self, symbol):
        """
        Get earnings calendar data
        
        Args:
            symbol: Stock symbol
        
        Returns:
            pandas.DataFrame: Earnings calendar
        """
        try:
            ticker = yf.Ticker(symbol)
            earnings = ticker.calendar
            
            return earnings
            
        except Exception as e:
            print(f"Warning: Could not fetch earnings data for {symbol}: {str(e)}")
            return None
    
    def search_stocks(self, query, limit=10):
        """
        Search for stocks by name or symbol
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            list: List of matching stocks
        """
        # This is a simplified search - in practice, you might want to use
        # a more comprehensive API for stock search
        popular_stocks = self.get_popular_stocks()
        
        results = []
        query_lower = query.lower()
        
        for symbol, info in popular_stocks.items():
            if (query_lower in symbol.lower() or 
                query_lower in info.get('company_name', '').lower()):
                results.append({
                    'symbol': symbol,
                    'name': info.get('company_name', 'N/A'),
                    'sector': info.get('sector', 'N/A')
                })
                
                if len(results) >= limit:
                    break
        
        return results
