"""
Machine Learning Model Module for Stock Forecasting Application
Includes LSTM, XGBoost, and ensemble learning with hyperparameter optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

class StockForecastingModel:
    """Advanced stock forecasting model with ensemble learning"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.lstm_model = None
        self.xgb_model = None
        self.ensemble_model = None
        self.feature_scalers = {}
        self.target_scaler = None
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, data):
        """
        Prepare features for machine learning
        
        Args:
            data (pandas.DataFrame): Stock data with technical indicators
            
        Returns:
            pandas.DataFrame: Prepared features
        """
        features = data.copy()
        
        # Select relevant features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Percent',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_12', 'EMA_20', 'EMA_26', 'EMA_50',
            'ATR', 'ADX', 'ADX_Pos', 'ADX_Neg',
            'Volume_SMA_10', 'Volume_SMA_20', 'OBV', 'CMF', 'Volume_Oscillator',
            'Price_Change', 'Volatility_10D', 'Volatility_20D'
        ]
        
        # Filter existing columns
        available_columns = [col for col in feature_columns if col in features.columns]
        features = features[available_columns].copy()
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Add lag features
        for lag in [1, 2, 3, 5, 10]:
            features[f'Close_lag_{lag}'] = features['Close'].shift(lag)
            features[f'Volume_lag_{lag}'] = features['Volume'].shift(lag)
            if 'RSI' in features.columns:
                features[f'RSI_lag_{lag}'] = features['RSI'].shift(lag)
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            features[f'Close_rolling_mean_{window}'] = features['Close'].rolling(window).mean()
            features[f'Close_rolling_std_{window}'] = features['Close'].rolling(window).std()
            features[f'Volume_rolling_mean_{window}'] = features['Volume'].rolling(window).mean()
        
        # Add technical ratios
        if all(col in features.columns for col in ['High', 'Low', 'Close']):
            features['HL_ratio'] = (features['High'] - features['Low']) / features['Close']
            features['OC_ratio'] = (features['Open'] - features['Close']) / features['Close']
        
        # Remove rows with NaN values
        features = features.dropna()
        
        self.feature_names = features.columns.tolist()
        return features
    
    def create_sequences(self, data, target_column='Close'):
        """
        Create sequences for LSTM training
        
        Args:
            data (pandas.DataFrame): Feature data
            target_column (str): Target column name
            
        Returns:
            tuple: (X_sequences, y_sequences)
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data.iloc[i-self.sequence_length:i].values)
            y.append(data[target_column].iloc[i])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape, optimize_params=None):
        """
        Build LSTM model architecture
        
        Args:
            input_shape (tuple): Input shape for LSTM
            optimize_params (dict): Hyperparameters for optimization
            
        Returns:
            tensorflow.keras.Model: LSTM model
        """
        if optimize_params is None:
            optimize_params = {
                'lstm_units_1': 128,
                'lstm_units_2': 64,
                'dropout_rate': 0.2,
                'learning_rate': 0.001
            }
        
        model = Sequential([
            LSTM(optimize_params['lstm_units_1'], return_sequences=True, input_shape=input_shape),
            Dropout(optimize_params['dropout_rate']),
            LSTM(optimize_params['lstm_units_2'], return_sequences=False),
            Dropout(optimize_params['dropout_rate']),
            Dense(50, activation='relu'),
            Dropout(optimize_params['dropout_rate']/2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=optimize_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, data, target_column='Close', validation_split=0.2, optimize_hyperparams=False):
        """
        Train the ensemble model
        
        Args:
            data (pandas.DataFrame): Training data with features
            target_column (str): Target column name
            validation_split (float): Validation split ratio
            optimize_hyperparams (bool): Whether to optimize hyperparameters
            
        Returns:
            dict: Training metrics
        """
        print("Preparing features...")
        features = self.prepare_features(data)
        
        if len(features) < self.sequence_length + 100:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length + 100}, got {len(features)}")
        
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        train_data = features.iloc[:split_idx]
        val_data = features.iloc[split_idx:]
        
        # Scale features
        print("Scaling features...")
        feature_scaler = StandardScaler()
        target_scaler = MinMaxScaler()
        
        # Fit scalers on training data
        train_features_scaled = feature_scaler.fit_transform(train_data)
        train_target_scaled = target_scaler.fit_transform(train_data[target_column].values.reshape(-1, 1)).flatten()
        
        val_features_scaled = feature_scaler.transform(val_data)
        val_target_scaled = target_scaler.transform(val_data[target_column].values.reshape(-1, 1)).flatten()
        
        self.feature_scalers['features'] = feature_scaler
        self.target_scaler = target_scaler
        
        # Prepare data for different models
        train_df_scaled = pd.DataFrame(train_features_scaled, columns=features.columns)
        val_df_scaled = pd.DataFrame(val_features_scaled, columns=features.columns)
        
        # LSTM preparation
        print("Preparing LSTM data...")
        X_lstm_train, y_lstm_train = self.create_sequences(train_df_scaled, target_column)
        X_lstm_val, y_lstm_val = self.create_sequences(val_df_scaled, target_column)
        
        # XGBoost preparation (use recent data points)
        X_xgb_train = train_features_scaled[self.sequence_length:]
        y_xgb_train = train_target_scaled[self.sequence_length:]
        X_xgb_val = val_features_scaled[self.sequence_length:]
        y_xgb_val = val_target_scaled[self.sequence_length:]
        
        # Train LSTM
        print("Training LSTM model...")
        best_lstm_params = {
            'lstm_units_1': 128, 'lstm_units_2': 64, 'dropout_rate': 0.2,
            'learning_rate': 0.001, 'batch_size': 32, 'epochs': 50
        }
        
        self.lstm_model = self.build_lstm_model(X_lstm_train.shape[1:], best_lstm_params)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        lstm_history = self.lstm_model.fit(
            X_lstm_train, y_lstm_train,
            validation_data=(X_lstm_val, y_lstm_val),
            epochs=best_lstm_params.get('epochs', 50),
            batch_size=best_lstm_params.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        # Train XGBoost
        print("Training XGBoost model...")
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.xgb_model.fit(X_xgb_train, y_xgb_train)
        
        # Evaluate models
        print("Evaluating models...")
        metrics = self.evaluate_models(X_lstm_val, y_lstm_val, X_xgb_val, y_xgb_val)
        
        self.is_trained = True
        
        return {
            'lstm_history': lstm_history.history,
            'metrics': metrics,
            'best_lstm_params': best_lstm_params
        }
    
    def evaluate_models(self, X_lstm_val, y_lstm_val, X_xgb_val, y_xgb_val):
        """
        Evaluate individual models and ensemble
        
        Args:
            X_lstm_val, y_lstm_val: LSTM validation data
            X_xgb_val, y_xgb_val: XGBoost validation data
            
        Returns:
            dict: Evaluation metrics
        """
        # LSTM predictions
        lstm_pred_scaled = self.lstm_model.predict(X_lstm_val, verbose=0)
        lstm_pred = self.target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
        y_lstm_true = self.target_scaler.inverse_transform(y_lstm_val.reshape(-1, 1)).flatten()
        
        # XGBoost predictions
        xgb_pred_scaled = self.xgb_model.predict(X_xgb_val)
        xgb_pred = self.target_scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).flatten()
        y_xgb_true = self.target_scaler.inverse_transform(y_xgb_val.reshape(-1, 1)).flatten()
        
        # Ensure same length for ensemble
        min_len = min(len(lstm_pred), len(xgb_pred))
        lstm_pred = lstm_pred[-min_len:]
        xgb_pred = xgb_pred[-min_len:]
        y_true = y_lstm_true[-min_len:]
        
        # Ensemble prediction (weighted average)
        ensemble_pred = 0.6 * lstm_pred + 0.4 * xgb_pred
        
        def calculate_metrics(y_true, y_pred, model_name):
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            return {
                f'{model_name}_MSE': mse,
                f'{model_name}_RMSE': rmse,
                f'{model_name}_MAE': mae,
                f'{model_name}_R2': r2,
                f'{model_name}_MAPE': mape
            }
        
        metrics = {}
        metrics.update(calculate_metrics(y_true, lstm_pred, 'LSTM'))
        metrics.update(calculate_metrics(y_true, xgb_pred, 'XGBoost'))
        metrics.update(calculate_metrics(y_true, ensemble_pred, 'Ensemble'))
        
        return metrics
    
    def predict(self, data, forecast_days=30):
        """
        Make predictions for future stock prices
        
        Args:
            data (pandas.DataFrame): Recent data for prediction
            forecast_days (int): Number of days to forecast
            
        Returns:
            dict: Predictions with confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        features = self.prepare_features(data)
        features_scaled = self.feature_scalers['features'].transform(features)
        features_df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
        
        predictions = []
        confidence_intervals = []
        
        # Use the last sequence_length points for initial prediction
        current_sequence = features_df_scaled.iloc[-self.sequence_length:].copy()
        current_features = features_scaled[-1:]
        
        for day in range(forecast_days):
            # LSTM prediction
            lstm_input = current_sequence.values.reshape(1, self.sequence_length, -1)
            lstm_pred_scaled = self.lstm_model.predict(lstm_input, verbose=0)[0, 0]
            
            # XGBoost prediction
            xgb_pred_scaled = self.xgb_model.predict(current_features)[0]
            
            # Ensemble prediction
            ensemble_pred_scaled = 0.6 * lstm_pred_scaled + 0.4 * xgb_pred_scaled
            
            # Inverse transform
            ensemble_pred = self.target_scaler.inverse_transform([[ensemble_pred_scaled]])[0, 0]
            predictions.append(ensemble_pred)
            
            # Calculate confidence interval (simple approach using model variance)
            lstm_pred = self.target_scaler.inverse_transform([[lstm_pred_scaled]])[0, 0]
            xgb_pred = self.target_scaler.inverse_transform([[xgb_pred_scaled]])[0, 0]
            
            model_std = np.std([lstm_pred, xgb_pred])
            confidence_interval = {
                'lower': ensemble_pred - 1.96 * model_std,
                'upper': ensemble_pred + 1.96 * model_std
            }
            confidence_intervals.append(confidence_interval)
            
            # Update sequence for next prediction (simplified)
            new_row = current_sequence.iloc[-1:].copy()
            new_row['Close'] = ensemble_pred_scaled
            
            # Append new row and remove oldest
            current_sequence = pd.concat([current_sequence.iloc[1:], new_row], ignore_index=True)
            current_features = new_row.values.reshape(1, -1)
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'forecast_days': forecast_days
        }
