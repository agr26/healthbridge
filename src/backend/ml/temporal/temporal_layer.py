import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class MedicalTemporalLayer(layers.Layer):
    """
    Custom layer for processing medical temporal data with attention.
    Handles irregular time intervals and missing data.
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MedicalTemporalLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Time encoding weights
        self.time_kernel = self.add_weight(
            name='time_kernel',
            shape=(1, self.output_dim),
            initializer='uniform',
            trainable=True
        )
        
        # Value encoding weights
        self.value_kernel = self.add_weight(
            name='value_kernel',
            shape=(input_shape[-1], self.output_dim),
            initializer='uniform',
            trainable=True
        )
        
        # Attention weights
        self.attention_weights = self.add_weight(
            name='attention_weights',
            shape=(self.output_dim, 1),
            initializer='uniform',
            trainable=True
        )
        
        super(MedicalTemporalLayer, self).build(input_shape)

    def call(self, inputs, timestamps=None, mask=None):
        # Handle inputs
        values, times = inputs
        
        # Encode time intervals
        time_encoding = tf.matmul(
            tf.expand_dims(times, -1),
            self.time_kernel
        )
        
        # Encode values
        value_encoding = tf.matmul(values, self.value_kernel)
        
        # Combine encodings
        combined_encoding = value_encoding + time_encoding
        
        # Apply attention
        attention_logits = tf.matmul(combined_encoding, self.attention_weights)
        attention_weights = tf.nn.softmax(attention_logits, axis=1)
        
        # Apply mask if provided
        if mask is not None:
            attention_weights *= tf.cast(mask, tf.float32)
            attention_weights /= tf.reduce_sum(
                attention_weights, 
                axis=1, 
                keepdims=True
            )
        
        # Weighted sum
        weighted_sum = tf.reduce_sum(
            combined_encoding * attention_weights, 
            axis=1
        )
        
        return weighted_sum

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# Usage example:
"""
# Create temporal layer
temporal_layer = MedicalTemporalLayer(32)

# Process data
values = tf.random.normal((batch_size, time_steps, features))
times = tf.range(time_steps, dtype=tf.float32)
mask = tf.ones((batch_size, time_steps))

# Apply layer
output = temporal_layer([values, times], mask=mask)
"""