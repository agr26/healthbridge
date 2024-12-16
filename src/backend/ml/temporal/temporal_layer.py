import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

@dataclass
class TemporalFeature:
    """Represents a temporal feature with its attributes"""
    name: str
    frequency: str  # 'hourly', 'daily', 'weekly', etc.
    interpolation_method: str  # 'linear', 'forward', 'backward'
    missing_threshold: float  # max acceptable missing data ratio
    normalization: str  # 'standard', 'minmax', 'robust'

@dataclass
class TemporalWindow:
    """Defines a temporal analysis window"""
    start_time: datetime
    end_time: datetime
    granularity: str
    features: List[TemporalFeature]

class MedicalTemporalLayer(layers.Layer):
    """
    Custom layer for processing medical temporal data with advanced features.
    Handles irregular time intervals, missing data, and multiple temporal scales.
    """
    
    def __init__(self, 
                 output_dim: int,
                 window_size: int = 48,
                 attention_heads: int = 4,
                 dropout_rate: float = 0.2,
                 **kwargs):
        """
        Initialize the Medical Temporal Layer.
        
        Args:
            output_dim: Dimension of output features
            window_size: Size of temporal window in hours
            attention_heads: Number of attention heads
            dropout_rate: Dropout rate for regularization
        """
        super(MedicalTemporalLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.window_size = window_size
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.logger = self._setup_logging()

    def build(self, input_shape):
        """Build the layer weights."""
        # Time encoding weights
        self.time_kernel = self.add_weight(
            name='time_kernel',
            shape=(1, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Value encoding weights
        self.value_kernel = self.add_weight(
            name='value_kernel',
            shape=(input_shape[-1], self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Multi-head attention weights
        self.attention_weights = [
            self.add_weight(
                name=f'attention_head_{i}',
                shape=(self.output_dim, self.output_dim),
                initializer='glorot_uniform',
                trainable=True
            )
            for i in range(self.attention_heads)
        ]
        
        # Gating mechanism weights
        self.gate_weights = self.add_weight(
            name='gate_weights',
            shape=(self.output_dim * 2, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(MedicalTemporalLayer, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], 
            training: bool = False, 
            mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Process temporal medical data.
        
        Args:
            inputs: Tuple of (values, times)
            training: Whether in training mode
            mask: Optional mask for missing values
            
        Returns:
            Processed temporal features
        """
        values, times = inputs
        batch_size = tf.shape(values)[0]
        
        # Handle missing values
        if mask is not None:
            values = tf.where(mask, values, tf.zeros_like(values))
        
        # Encode time intervals
        time_encoding = self._encode_time(times)
        
        # Encode values
        value_encoding = tf.matmul(values, self.value_kernel)
        
        # Apply multi-head attention
        attention_outputs = []
        for head in range(self.attention_heads):
            attention = self._compute_attention(
                value_encoding,
                time_encoding,
                self.attention_weights[head]
            )
            attention_outputs.append(attention)
        
        # Combine attention heads
        multi_head = tf.concat(attention_outputs, axis=-1)
        
        # Apply gating mechanism
        gate_input = tf.concat([value_encoding, multi_head], axis=-1)
        gate = tf.sigmoid(tf.matmul(gate_input, self.gate_weights))
        
        # Final output
        output = gate * value_encoding + (1 - gate) * multi_head
        
        # Apply dropout during training
        if training:
            output = tf.nn.dropout(output, rate=self.dropout_rate)
        
        return output

    def _encode_time(self, times: tf.Tensor) -> tf.Tensor:
        """
        Encode time intervals using positional encoding.
        
        Args:
            times: Tensor of time values
            
        Returns:
            Encoded time features
        """
        # Convert to relative time differences
        time_diffs = times[:, 1:] - times[:, :-1]
        
        # Apply positional encoding
        position = tf.range(0, self.output_dim, 2, dtype=tf.float32)
        div_term = tf.exp(position * (-tf.math.log(10000.0) / self.output_dim))
        
        # Compute sin/cos encoding
        pe_sin = tf.sin(tf.expand_dims(time_diffs, -1) * div_term)
        pe_cos = tf.cos(tf.expand_dims(time_diffs, -1) * div_term)
        
        return tf.concat([pe_sin, pe_cos], axis=-1)

    def _compute_attention(self, 
                         query: tf.Tensor,
                         key: tf.Tensor,
                         weight: tf.Tensor) -> tf.Tensor:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            weight: Attention weight matrix
            
        Returns:
            Attention output
        """
        # Compute attention scores
        scaled_attention_logits = tf.matmul(query, key, transpose_b=True)
        scaled_attention_logits /= tf.math.sqrt(tf.cast(self.output_dim, tf.float32))
        
        # Apply softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply attention
        return tf.matmul(attention_weights, tf.matmul(key, weight))

    def process_temporal_window(self, 
                              window: TemporalWindow,
                              data: Dict[str, np.ndarray]) -> tf.Tensor:
        """
        Process a temporal window of medical data.
        
        Args:
            window: TemporalWindow configuration
            data: Dictionary of temporal features
            
        Returns:
            Processed temporal features
        """
        try:
            # Validate data
            self._validate_temporal_data(window, data)
            
            # Align temporal features
            aligned_data = self._align_temporal_features(window, data)
            
            # Handle missing values
            interpolated_data = self._handle_missing_values(
                aligned_data,
                window.features
            )
            
            # Convert to tensor
            values = tf.convert_to_tensor(interpolated_data, dtype=tf.float32)
            times = self._generate_time_points(window)
            
            # Process through layer
            return self.call((values, times))
            
        except Exception as e:
            self.logger.error(f"Error processing temporal window: {str(e)}")
            raise

    def _validate_temporal_data(self, 
                              window: TemporalWindow,
                              data: Dict[str, np.ndarray]) -> None:
        """Validate temporal data against window configuration."""
        for feature in window.features:
            if feature.name not in data:
                raise ValueError(f"Missing feature: {feature.name}")
            
            missing_ratio = np.isnan(data[feature.name]).mean()
            if missing_ratio > feature.missing_threshold:
                raise ValueError(
                    f"Too many missing values for feature {feature.name}: "
                    f"{missing_ratio:.2%}"
                )

    def _align_temporal_features(self,
                               window: TemporalWindow,
                               data: Dict[str, np.ndarray]) -> np.ndarray:
        """Align temporal features to common time points."""
        aligned_data = []
        time_points = self._generate_time_points(window)
        
        for feature in window.features:
            # Interpolate to common time points
            interpolated = self._interpolate_feature(
                data[feature.name],
                time_points,
                feature.interpolation_method
            )
            aligned_data.append(interpolated)
            
        return np.stack(aligned_data, axis=-1)

    def _handle_missing_values(self,
                             data: np.ndarray,
                             features: List[TemporalFeature]) -> np.ndarray:
        """Handle missing values in temporal data."""
        for i, feature in enumerate(features):
            if feature.interpolation_method == 'forward':
                data[:, i] = self._forward_fill(data[:, i])
            elif feature.interpolation_method == 'backward':
                data[:, i] = self._backward_fill(data[:, i])
            elif feature.interpolation_method == 'linear':
                data[:, i] = self._linear_interpolate(data[:, i])
                
        return data

    @staticmethod
    def _forward_fill(data: np.ndarray) -> np.ndarray:
        """Forward fill missing values."""
        mask = np.isnan(data)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        np.maximum.accumulate(idx, out=idx)
        return data[idx]

    @staticmethod
    def _backward_fill(data: np.ndarray) -> np.ndarray:
        """Backward fill missing values."""
        return np.flip(MedicalTemporalLayer._forward_fill(np.flip(data)))

    @staticmethod
    def _linear_interpolate(data: np.ndarray) -> np.ndarray:
        """Linear interpolation of missing values."""
        if np.all(np.isnan(data)):
            return np.zeros_like(data)
        return np.interp(
            np.arange(len(data)),
            np.arange(len(data))[~np.isnan(data)],
            data[~np.isnan(data)]
        )

    def _generate_time_points(self, window: TemporalWindow) -> np.ndarray:
        """Generate regular time points for the window."""
        if window.granularity == 'hourly':
            delta = timedelta(hours=1)
        elif window.granularity == 'daily':
            delta = timedelta(days=1)
        else:
            raise ValueError(f"Unsupported granularity: {window.granularity}")
            
        time_points = []
        current = window.start_time
        while current <= window.end_time:
            time_points.append(current)
            current += delta
            
        return np.array([(t - window.start_time).total_seconds() 
                        for t in time_points])

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the temporal layer."""
        logger = logging.getLogger('MedicalTemporal')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('logs/temporal_layer.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_config(self) -> Dict:
        """Get layer configuration."""
        config = super(MedicalTemporalLayer, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'window_size': self.window_size,
            'attention_heads': self.attention_heads,
            'dropout_rate': self.dropout_rate
        })
        return config