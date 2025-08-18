# Stock-Market-Analysis
Overview
The Stock Market Dashboard is a clean, user-friendly Streamlit web application focused on visual stock market analysis and company information display. The platform has been simplified from its original ML-focused architecture to provide clear, graphical presentation of stock data with interactive charts and comprehensive company search functionality. Built with a streamlined interface, the application offers real-time data fetching via the Yahoo Finance API and beautiful visualizations through Plotly candlestick charts.

User Preferences
Preferred communication style: Simple, everyday language. Focus: Clean visual presentation of company stock values in graphical format without complex ML features. Company Selection: Expanded comprehensive list covering multiple sectors for better search functionality.

System Architecture
Frontend Architecture
Framework: Streamlit for web interface with multi-page application structure
Visualization: Plotly for interactive charts and graphs with custom styling
Layout: Wide layout configuration with expandable sidebar navigation
Pages: Modular page structure with separate files for each major feature (Stock Prediction, Technical Analysis, Portfolio Optimization, Risk Assessment, Model Performance)
Backend Architecture
ML Models: Dual model approach using TensorFlow/Keras LSTM networks for time series prediction and scikit-learn Random Forest for feature-based prediction
Data Processing: Pandas and NumPy for data manipulation with custom preprocessing pipelines
Model Architecture: Object-oriented design with separate classes for each model type, supporting training, prediction, and performance evaluation
Feature Engineering: Comprehensive technical indicator calculation including moving averages, RSI, MACD, Bollinger Bands, and volatility metrics
Data Management
Data Source: Yahoo Finance API through yfinance library for real-time and historical stock data
Caching Strategy: Built-in retry mechanisms and error handling for API reliability
Data Pipeline: Automated data cleaning, feature extraction, and preprocessing workflows
Storage: In-memory data processing with model persistence through joblib
Analytics Engine
Portfolio Optimization: Modern Portfolio Theory implementation using scipy optimization for efficient frontier calculation
Risk Assessment: Comprehensive risk metrics including Value at Risk (VaR), Conditional VaR, Sharpe ratios, and drawdown analysis
Technical Analysis: Complete suite of technical indicators with ML-enhanced signal generation
Performance Metrics: Model evaluation using MSE, MAE, R² score, and directional accuracy measurements
Model Training and Prediction
LSTM Architecture: Multi-layer LSTM networks with dropout regularization for sequence prediction
Random Forest: Ensemble learning with feature importance analysis and hyperparameter optimization
Cross-Validation: Time series cross-validation for robust model evaluation
Feature Selection: Automated feature engineering with correlation analysis and importance ranking
External Dependencies
Data Providers
Yahoo Finance API: Primary data source for stock prices, financial metrics, and company information via yfinance library
Market Data: Real-time and historical price data, volume, dividends, and stock splits
Machine Learning Libraries
TensorFlow/Keras: Deep learning framework for LSTM neural network implementation
scikit-learn: Machine learning library for Random Forest models, preprocessing, and evaluation metrics
scipy: Scientific computing library for optimization algorithms and statistical functions
Visualization and UI
Streamlit: Web application framework for the user interface and interactive components
Plotly: Interactive plotting library for financial charts, technical indicators, and performance visualizations
Data Processing
pandas: Data manipulation and analysis library for time series operations
NumPy: Numerical computing library for mathematical operations and array processing
Utility Libraries
joblib: Model serialization and persistence for trained ML models
warnings: Python warnings management for clean output
datetime: Date and time manipulation for time series analysis
Optional Integrations
Custom Technical Indicators: Extensible framework for adding new technical analysis tools
Risk Management: Advanced risk calculation methods including Monte Carlo simulations
Portfolio Analytics: Performance attribution and factor analysis capabilities
