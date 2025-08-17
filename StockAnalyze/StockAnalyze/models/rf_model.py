import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

class RandomForestStockPredictor:
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Initialize Random Forest Stock Predictor
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
    def create_features(self, df, target_column='Close'):
        """
        Create features from stock data including technical indicators
        """
        features_df = df.copy()
        
        # Price-based features
        features_df['Returns'] = features_df[target_column].pct_change()
        features_df['Log_Returns'] = np.log(features_df[target_column] / features_df[target_column].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features_df[f'MA_{window}'] = features_df[target_column].rolling(window=window).mean()
            features_df[f'MA_ratio_{window}'] = features_df[target_column] / features_df[f'MA_{window}']
        
        # Volatility features
        for window in [5, 10, 20]:
            features_df[f'Volatility_{window}'] = features_df['Returns'].rolling(window=window).std()
        
        # Price position features
        features_df['High_Low_Ratio'] = features_df['High'] / features_df['Low']
        features_df['Close_Open_Ratio'] = features_df[target_column] / features_df['Open']
        
        # Volume features
        features_df['Volume_MA_5'] = features_df['Volume'].rolling(window=5).mean()
        features_df['Volume_Ratio'] = features_df['Volume'] / features_df['Volume_MA_5']
        
        # Bollinger Bands
        bb_window = 20
        bb_ma = features_df[target_column].rolling(window=bb_window).mean()
        bb_std = features_df[target_column].rolling(window=bb_window).std()
        features_df['BB_Upper'] = bb_ma + (bb_std * 2)
        features_df['BB_Lower'] = bb_ma - (bb_std * 2)
        features_df['BB_Position'] = (features_df[target_column] - features_df['BB_Lower']) / (features_df['BB_Upper'] - features_df['BB_Lower'])
        
        # RSI (Relative Strength Index)
        delta = features_df[target_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = features_df[target_column].ewm(span=12).mean()
        exp2 = features_df[target_column].ewm(span=26).mean()
        features_df['MACD'] = exp1 - exp2
        features_df['MACD_Signal'] = features_df['MACD'].ewm(span=9).mean()
        features_df['MACD_Histogram'] = features_df['MACD'] - features_df['MACD_Signal']
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features_df[f'Close_Lag_{lag}'] = features_df[target_column].shift(lag)
            features_df[f'Volume_Lag_{lag}'] = features_df['Volume'].shift(lag)
            features_df[f'Returns_Lag_{lag}'] = features_df['Returns'].shift(lag)
        
        # Time-based features
        features_df['DayOfWeek'] = features_df.index.dayofweek
        features_df['Month'] = features_df.index.month
        features_df['Quarter'] = features_df.index.quarter
        
        # Drop original OHLCV columns and target
        cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'] if 'Adj Close' in features_df.columns else ['Open', 'High', 'Low', 'Close', 'Volume']
        features_df = features_df.drop(columns=cols_to_drop)
        
        return features_df
    
    def prepare_data(self, df, target_column='Close', look_ahead=1):
        """
        Prepare features and target for training
        """
        # Create features
        features_df = self.create_features(df, target_column)
        
        # Create target (future price)
        target = df[target_column].shift(-look_ahead)
        
        # Align indices and drop NaN values
        aligned_df = pd.concat([features_df, target.rename('target')], axis=1)
        aligned_df = aligned_df.dropna()
        
        X = aligned_df.drop(columns=['target'])
        y = aligned_df['target']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, df, target_column='Close', test_size=0.2, optimize_hyperparams=False):
        """
        Train the Random Forest model
        """
        # Prepare data
        X, y = self.prepare_data(df, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if optimize_hyperparams:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Use default parameters
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
            best_params = None
        
        self.is_trained = True
        
        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'best_params': best_params
        }
        
        return metrics, (X_train, X_test, y_train, y_test, train_pred, test_pred)
    
    def predict(self, df, target_column='Close'):
        """
        Make predictions on new data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X, _ = self.prepare_data(df, target_column)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return pd.Series(predictions, index=X.index)
    
    def predict_future(self, df, days=30, target_column='Close'):
        """
        Predict future stock prices using iterative approach
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_df = df.copy()
        
        for day in range(days):
            # Prepare features for the last available data point
            X, _ = self.prepare_data(current_df, target_column)
            
            if len(X) == 0:
                break
            
            # Scale and predict
            X_last = X.iloc[[-1]]
            X_scaled = self.scaler.transform(X_last)
            pred = self.model.predict(X_scaled)[0]
            
            predictions.append(pred)
            
            # Add predicted value to dataframe for next iteration
            last_date = current_df.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            
            # Create a new row with predicted close price
            new_row = current_df.iloc[-1].copy()
            new_row[target_column] = pred
            new_row['Open'] = pred  # Simplified assumption
            new_row['High'] = pred * 1.02  # Simplified assumption
            new_row['Low'] = pred * 0.98   # Simplified assumption
            
            # Add to dataframe
            current_df.loc[next_date] = new_row
        
        # Create future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions), freq='D')
        
        return pd.Series(predictions, index=future_dates)
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def analyze_predictions(self, df, target_column='Close'):
        """
        Analyze prediction accuracy and patterns
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        predictions = self.predict(df, target_column)
        actual = df[target_column].reindex(predictions.index)
        
        # Calculate various accuracy metrics
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        # Direction accuracy (up/down prediction)
        actual_direction = (actual.diff() > 0).astype(int)
        pred_direction = (predictions.diff() > 0).astype(int)
        direction_accuracy = (actual_direction == pred_direction).mean()
        
        analysis = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'direction_accuracy': direction_accuracy,
            'predictions': predictions,
            'actual': actual
        }
        
        return analysis
    
    def save_model(self, filepath):
        """
        Save the trained model and scaler
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f"{filepath}_model.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        # Save feature names and parameters
        metadata = {
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
    
    def load_model(self, filepath):
        """
        Load a trained model and scaler
        """
        # Load model
        self.model = joblib.load(f"{filepath}_model.pkl")
        
        # Load scaler
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        
        # Load metadata
        metadata = joblib.load(f"{filepath}_metadata.pkl")
        self.feature_names = metadata['feature_names']
        self.n_estimators = metadata['n_estimators']
        self.max_depth = metadata['max_depth']
        self.random_state = metadata['random_state']
        
        self.is_trained = True
