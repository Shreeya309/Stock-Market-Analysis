import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

class LSTMStockPredictor:
    def __init__(self, sequence_length=60, units=50, dropout=0.2):
        """
        Initialize LSTM Stock Predictor
        
        Args:
            sequence_length: Number of time steps to look back
            units: Number of LSTM units
            dropout: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def create_sequences(self, data, target_col='Close'):
        """
        Create sequences for LSTM training
        """
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(data)):
            sequences.append(data[i-self.sequence_length:i])
            targets.append(data[i][target_col] if isinstance(data, pd.DataFrame) else data[i])
            
        return np.array(sequences), np.array(targets)
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        """
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(self.units, return_sequences=True),
            Dropout(self.dropout),
            LSTM(self.units, return_sequences=False),
            Dropout(self.dropout),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, df, target_column='Close'):
        """
        Prepare data for LSTM training
        """
        # Select features
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = df[features].copy()
        
        # Fill missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        scaled_df = pd.DataFrame(scaled_data, columns=features, index=data.index)
        
        # Create sequences
        X, y = self.create_sequences(scaled_df.values, features.index(target_column))
        
        return X, y, scaled_df
    
    def train(self, df, target_column='Close', test_size=0.2, epochs=50, batch_size=32, verbose=1):
        """
        Train the LSTM model
        """
        # Prepare data
        X, y, scaled_df = self.prepare_data(df, target_column)
        
        # Split data
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'history': history.history
        }
        
        return metrics, (X_train, X_test, y_train, y_test, train_pred, test_pred)
    
    def predict(self, data, steps=1):
        """
        Make predictions using the trained model
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_sequence = data[-self.sequence_length:].copy()
        
        for _ in range(steps):
            # Reshape for prediction
            sequence_reshaped = current_sequence.reshape(1, self.sequence_length, current_sequence.shape[1])
            
            # Make prediction
            pred = self.model.predict(sequence_reshaped, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            new_row = current_sequence[-1].copy()
            new_row[3] = pred[0, 0]  # Assuming Close is at index 3
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(predictions)
    
    def predict_future(self, df, days=30):
        """
        Predict future stock prices
        """
        # Prepare the last sequence of data
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = df[features].copy().fillna(method='ffill').fillna(method='bfill')
        scaled_data = self.scaler.transform(data)
        
        # Make predictions
        predictions = self.predict(scaled_data, steps=days)
        
        # Inverse transform predictions (only for Close price)
        dummy_array = np.zeros((len(predictions), 5))
        dummy_array[:, 3] = predictions  # Close price is at index 3
        inverse_pred = self.scaler.inverse_transform(dummy_array)[:, 3]
        
        # Create future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        return pd.Series(inverse_pred, index=future_dates)
    
    def get_feature_importance(self, df, target_column='Close'):
        """
        Analyze feature importance using permutation
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X, y, _ = self.prepare_data(df, target_column)
        
        # Baseline performance
        baseline_pred = self.model.predict(X[-100:], verbose=0)  # Use last 100 samples
        baseline_mse = mean_squared_error(y[-100:], baseline_pred)
        
        importance_scores = {}
        
        for i, feature in enumerate(features):
            # Create permuted data
            X_permuted = X[-100:].copy()
            np.random.shuffle(X_permuted[:, :, i])
            
            # Calculate performance drop
            permuted_pred = self.model.predict(X_permuted, verbose=0)
            permuted_mse = mean_squared_error(y[-100:], permuted_pred)
            
            importance_scores[feature] = permuted_mse - baseline_mse
        
        return importance_scores
    
    def save_model(self, filepath):
        """
        Save the trained model and scaler
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(f"{filepath}_model.h5")
        
        # Save scaler
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        # Save parameters
        params = {
            'sequence_length': self.sequence_length,
            'units': self.units,
            'dropout': self.dropout
        }
        joblib.dump(params, f"{filepath}_params.pkl")
    
    def load_model(self, filepath):
        """
        Load a trained model and scaler
        """
        # Load model
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        
        # Load scaler
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        
        # Load parameters
        params = joblib.load(f"{filepath}_params.pkl")
        self.sequence_length = params['sequence_length']
        self.units = params['units']
        self.dropout = params['dropout']
        
        self.is_trained = True
