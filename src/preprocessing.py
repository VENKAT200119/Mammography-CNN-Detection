"""Image Preprocessing Module

This module handles preprocessing of mammography images including:
- Loading images from MIAS dataset
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Morphological operations
- Noise reduction
- Image enhancement
"""

import cv2
import numpy as np
from skimage import exposure
from scipy import ndimage


class MammographyPreprocessor:
    """Preprocessing pipeline for mammography images"""
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size for model input (default: 224x224 for GoogleNet)
        """
        self.target_size = target_size
    
    def load_image(self, image_path):
        """
        Load mammography image from file
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array: Loaded grayscale image
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return img
    
    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply CLAHE for contrast enhancement
        
        Args:
            image: Input grayscale image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            numpy array: Enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        return enhanced
    
    def remove_noise(self, image, kernel_size=5):
        """
        Remove noise using Gaussian blur
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel
            
        Returns:
            numpy array: Denoised image
        """
        denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return denoised
    
    def morphological_operations(self, image, operation='close', kernel_size=5):
        """
        Apply morphological operations
        
        Args:
            image: Input binary/grayscale image
            operation: Type of operation ('open', 'close', 'erode', 'dilate')
            kernel_size: Size of structuring element
            
        Returns:
            numpy array: Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'close':
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'erode':
            result = cv2.erode(image, kernel)
        elif operation == 'dilate':
            result = cv2.dilate(image, kernel)
        else:
            result = image
            
        return result
    
    def resize_image(self, image):
        """
        Resize image to target size
        
        Args:
            image: Input image
            
        Returns:
            numpy array: Resized image
        """
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        return resized
    
    def normalize(self, image):
        """
        Normalize image to [0, 1] range
        
        Args:
            image: Input image
            
        Returns:
            numpy array: Normalized image
        """
        normalized = image.astype(np.float32) / 255.0
        return normalized
    
    def preprocess_pipeline(self, image_path):
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to input image
            
        Returns:
            numpy array: Fully preprocessed image ready for model input
        """
        # Load image
        img = self.load_image(image_path)
        
        # Apply CLAHE for contrast enhancement
        img = self.apply_clahe(img)
        
        # Remove noise
        img = self.remove_noise(img)
        
        # Apply morphological closing
        img = self.morphological_operations(img, operation='close')
        
        # Resize to target size
        img = self.resize_image(img)
        
        # Normalize
        img = self.normalize(img)
        
        # Add channel dimension for grayscale
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        return img


def batch_preprocess(image_paths, target_size=(224, 224)):
    """
    Preprocess multiple images
    
    Args:
        image_paths: List of image file paths
        target_size: Target image size
        
    Returns:
        numpy array: Batch of preprocessed images
    """
    preprocessor = MammographyPreprocessor(target_size=target_size)
    processed_images = []
    
    for path in image_paths:
        try:
            img = preprocessor.preprocess_pipeline(path)
            processed_images.append(img)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    
    return np.array(processed_images)
