import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

class HealthcareNeuralNetwork:
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.history = {}
        
    def build_readmission_predictor(self, input_dim: int) -> tf.keras.Model:
        """
        Build neural network for readmission prediction.
        Incorporates clinical temporal patterns and patient characteristics.
        """
        # Input layers for different data types
        clinical_input = layers.Input(shape=(input_dim,), name='clinical_data')
        temporal_input = layers.Input(shape=(self.config['temporal_steps'], 
                                           self.config['temporal_features']), 
                                    name='temporal_data')
        
        # Process clinical data
        clinical_branch = layers.Dense(64, activation='relu', 
                                     kernel_regularizer=regularizers.l2(0.01))(clinical_input)
        clinical_branch = layers.Dropout(0.3)(clinical_branch)
        clinical_branch = layers.Dense(32, activation='relu')(clinical_branch)
        
        # Process temporal data using LSTM
        temporal_branch = layers.LSTM(32, return_sequences=True)(temporal_input)
        temporal_branch = layers.LSTM(16)(temporal_branch)
        
        # Combine branches
        combined = layers.Concatenate()([clinical_branch, temporal_branch])
        
        # Deep layers with residual connections
        deep = layers.Dense(32, activation='relu')(combined)
        residual = deep
        
        deep = layers.Dense(32, activation='relu')(deep)
        deep = layers.Add()([deep, residual])  # Residual connection
        
        deep = layers.Dense(16, activation='relu')(deep)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='readmission_risk')(deep)
        
        # Create model
        model = models.Model(
            inputs=[clinical_input, temporal_input],
            outputs=output
        )
        
        # Compile model with custom metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model

    def build_los_predictor(self, input_dim: int) -> tf.keras.Model:
        """
        Build neural network for length of stay prediction.
        Uses attention mechanism for temporal data processing.
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),
            
            # Embedding layer for categorical variables
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Attention mechanism
            layers.Dense(64, activation='relu'),
            layers.Attention(),  # Self-attention layer
            layers.Dropout(0.2),
            
            # Deep layers
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            
            # Output layer
            layers.Dense(1, activation='linear')  # Linear activation for LOS prediction
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model

    def build_complication_predictor(self, input_dim: int) -> tf.keras.Model:
        """
        Build neural network for predicting medical complications.
        Uses multi-head attention for feature interaction learning.
        """
        inputs = layers.Input(shape=(input_dim,))
        
        # Multi-head attention
        attention_heads = []
        for _ in range(4):  # 4 attention heads
            attention_head = layers.Dense(32, activation='relu')(inputs)
            attention_head = layers.Attention()(
                [attention_head, attention_head, attention_head]
            )
            attention_heads.append(attention_head)
        
        # Combine attention heads
        merged = layers.Concatenate()(attention_heads)
        
        # Deep network
        x = layers.Dense(64, activation='relu')(merged)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Multi-task outputs
        outputs = []
        for complication in self.config['complications']:
            output = layers.Dense(1, activation='sigmoid', 
                                name=f'complication_{complication}')(x)
            outputs.append(output)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss=['binary_crossentropy' for _ in outputs],
            metrics=['accuracy', 'auc']
        )
        
        return model

    def train_model(self, model_type: str, X: np.ndarray, y: np.ndarray, 
                   validation_split: float = 0.2) -> Dict:
        """
        Train the neural network with advanced techniques.
        """
        # Preprocess data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Select model
        if model_type == 'readmission':
            model = self.build_readmission_predictor(X.shape[1])
        elif model_type == 'los':
            model = self.build_los_predictor(X.shape[1])
        elif model_type == 'complications':
            model = self.build_complication_predictor(X.shape[1])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train with callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3
            )
        ]
        
        history = model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks
        )
        
        # Store model and results
        self.models[model_type] = model
        self.scalers[model_type] = scaler
        self.history[model_type] = history.history
        
        # Calculate feature importance
        self.feature_importance[model_type] = self._calculate_feature_importance(
            model, X_train_scaled
        )
        
        return {
            'history': history.history,
            'feature_importance': self.feature_importance[model_type]
        }

    def _calculate_feature_importance(self, model: tf.keras.Model, 
                                   X: np.ndarray) -> Dict:
        """
        Calculate feature importance using SHAP values.
        """
        explainer = shap.DeepExplainer(model, X[:100])  # Use subset for efficiency
        shap_values = explainer.shap_values(X[:100])
        
        # Calculate mean absolute SHAP values for each feature
        feature_importance = np.mean(np.abs(shap_values[0]), axis=0)
        
        return {
            'shap_values': shap_values,
            'importance_scores': feature_importance
        }

    def evaluate_model(self, model_type: str, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance with detailed metrics.
        """
        model = self.models[model_type]
        scaler = self.scalers[model_type]
        
        # Scale test data
        X_test_scaled = scaler.transform(X_test)
        
        # Get predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred),
            'predictions': y_pred,
            'feature_importance': self.feature_importance[model_type]
        }
        
        # Generate visualizations
        self._generate_evaluation_plots(
            y_test, 
            y_pred, 
            self.history[model_type],
            model_type
        )
        
        return metrics

    def _generate_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                history: Dict, model_type: str) -> None:
        """
        Generate evaluation plots for model performance.
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        axes[0, 0].plot(fpr, tpr)
        axes[0, 0].set_title('ROC Curve')
        
        # Training history
        axes[0, 1].plot(history['loss'], label='Training Loss')
        axes[0, 1].plot(history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Training History')
        axes[0, 1].legend()
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': range(len(self.feature_importance[model_type]['importance_scores'])),
            'importance': self.feature_importance[model_type]['importance_scores']
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=importance_df.head(10), x='importance', y='feature', ax=axes[1, 0])
        axes[1, 0].set_title('Top 10 Feature Importance')
        
        # Calibration plot
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred)
        axes[1, 1].plot(mean_predicted_value, fraction_of_positives)
        axes[1, 1].set_title('Calibration Plot')
        
        plt.tight_layout()
        plt.savefig(f'evaluation_{model_type}.png')
        plt.close()