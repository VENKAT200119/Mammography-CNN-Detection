"""Training and Evaluation Module

This module handles model training, evaluation, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns


def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        X: Feature data
        y: Labels (one-hot encoded)
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=np.argmax(y, axis=1))


def get_callbacks(model_path='models/best_model.h5'):
    """
    Get training callbacks
    
    Returns:
        List of callbacks
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    return [early_stop, checkpoint, reduce_lr]


def train_model(model, X_train, y_train, X_val, y_val, 
               epochs=100, batch_size=32, callbacks=None):
    """
    Train the model
    
    Args:
        model: Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of epochs
        batch_size: Batch size
        callbacks: List of callbacks
        
    Returns:
        History object
    """
    if callbacks is None:
        callbacks = get_callbacks()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, class_names=['Fatty', 'Glandular', 'Dense']):
    """
    Evaluate model performance
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Calculate per-class metrics
    sensitivity = np.diag(cm) / np.sum(cm, axis=1)
    specificity = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(tn / (tn + fp))
    
    metrics = {
        'accuracy': test_accuracy,
        'loss': test_loss,
        'sensitivity': np.mean(sensitivity),
        'specificity': np.mean(specificity),
        'confusion_matrix': cm
    }
    
    return metrics


def plot_training_history(history, save_path='results/training_history.png'):
    """
    Plot training history
    
    Args:
        history: Keras History object
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path='results/confusion_matrix.png'):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")
