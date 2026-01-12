"""Deep Learning Models Module

This module contains CNN architectures for mammography classification:
- GoogleNet (Inception v1) - 22 layers
- Custom CNN models
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3


def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3,
                    filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
    """
    Inception module for GoogleNet
    
    Args:
        x: Input tensor
        Various filter parameters for different paths
        name: Module name
        
    Returns:
        Concatenated output tensor
    """
    # 1x1 convolution path
    conv_1x1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu',
                            name=f'{name}_1x1' if name else None)(x)
    
    # 3x3 convolution path
    conv_3x3 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu',
                            name=f'{name}_3x3' if name else None)(conv_3x3)
    
    # 5x5 convolution path
    conv_5x5 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu',
                            name=f'{name}_5x5' if name else None)(conv_5x5)
    
    # MaxPooling path
    pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool = layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu',
                        name=f'{name}_pool_proj' if name else None)(pool)
    
    # Concatenate all paths
    output = layers.Concatenate(axis=-1, name=f'{name}_concat' if name else None)(
        [conv_1x1, conv_3x3, conv_5x5, pool])
    
    return output


def build_googlenet(input_shape=(224, 224, 1), num_classes=3):
    """
    Build GoogleNet (Inception v1) architecture
    
    Args:
        input_shape: Input image shape (default: 224x224x1 for grayscale)
        num_classes: Number of output classes (default: 3)
        
    Returns:
        Keras Model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolutions
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Inception 3a
    x = inception_module(x, 64, 96, 128, 16, 32, 32, name='inception_3a')
    # Inception 3b
    x = inception_module(x, 128, 128, 192, 32, 96, 64, name='inception_3b')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Inception 4a
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_4a')
    # Inception 4b
    x = inception_module(x, 160, 112, 224, 24, 64, 64, name='inception_4b')
    # Inception 4c
    x = inception_module(x, 128, 128, 256, 24, 64, 64, name='inception_4c')
    # Inception 4d
    x = inception_module(x, 112, 144, 288, 32, 64, 64, name='inception_4d')
    # Inception 4e
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_4e')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Inception 5a
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_5a')
    # Inception 5b
    x = inception_module(x, 384, 192, 384, 48, 128, 128, name='inception_5b')
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='GoogleNet')
    return model


def build_simple_cnn(input_shape=(224, 224, 1), num_classes=3):
    """
    Build a simpler CNN for comparison
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Keras Model
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='SimpleCNN')
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile model with Adam optimizer
    
    Args:
        model: Keras model
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    return model
