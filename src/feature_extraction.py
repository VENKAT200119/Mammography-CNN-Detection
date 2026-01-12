"""Feature Extraction Module

This module extracts features from preprocessed mammography images:
- DWT (Discrete Wavelet Transform)
- GLCM (Gray Level Co-occurrence Matrix)
- HOG (Histogram of Oriented Gradients)
"""

import numpy as np
import pywt
from skimage.feature import greycomatrix, greycoprops, hog
from skimage import exposure
import cv2


class FeatureExtractor:
    """Extract multiple features from mammography images"""
    
    def __init__(self, wavelet='db4', glcm_distances=[1], glcm_angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        Initialize feature extractor
        
        Args:
            wavelet: Wavelet type for DWT (default: 'db4')
            glcm_distances: Distances for GLCM calculation
            glcm_angles: Angles for GLCM calculation
        """
        self.wavelet = wavelet
        self.glcm_distances = glcm_distances
        self.glcm_angles = glcm_angles
    
    def extract_dwt_features(self, image):
        """
        Extract DWT features using 2D wavelet decomposition
        
        Args:
            image: Input grayscale image
            
        Returns:
            dict: Dictionary of DWT features (energy, entropy, mean, std)
        """
        # Perform 2-level DWT
        coeffs2 = pywt.dwt2(image, self.wavelet)
        LL, (LH, HL, HH) = coeffs2
        
        features = {}
        for name, coeff in [('LL', LL), ('LH', LH), ('HL', HL), ('HH', HH)]:
            features[f'dwt_{name}_mean'] = np.mean(coeff)
            features[f'dwt_{name}_std'] = np.std(coeff)
            features[f'dwt_{name}_energy'] = np.sum(coeff**2)
            features[f'dwt_{name}_entropy'] = self._calculate_entropy(coeff)
        
        return features
    
    def extract_glcm_features(self, image):
        """
        Extract GLCM texture features
        
        Args:
            image: Input grayscale image (0-255)
            
        Returns:
            dict: Dictionary of GLCM features
        """
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Calculate GLCM
        glcm = greycomatrix(image, distances=self.glcm_distances, 
                          angles=self.glcm_angles, levels=256, 
                          symmetric=True, normed=True)
        
        # Extract properties
        features = {}
        properties = ['contrast', 'dissimilarity', 'homogeneity', 
                     'energy', 'correlation', 'ASM']
        
        for prop in properties:
            values = greycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = np.mean(values)
            features[f'glcm_{prop}_std'] = np.std(values)
        
        return features
    
    def extract_hog_features(self, image):
        """
        Extract HOG (Histogram of Oriented Gradients) features
        
        Args:
            image: Input grayscale image
            
        Returns:
            dict: Dictionary of HOG features
        """
        # Calculate HOG features
        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=True, 
                           channel_axis=None)
        
        features = {
            'hog_mean': np.mean(fd),
            'hog_std': np.std(fd),
            'hog_max': np.max(fd),
            'hog_min': np.min(fd),
            'hog_energy': np.sum(fd**2)
        }
        
        return features, hog_image
    
    def extract_statistical_features(self, image):
        """
        Extract basic statistical features
        
        Args:
            image: Input image
            
        Returns:
            dict: Dictionary of statistical features
        """
        features = {
            'mean': np.mean(image),
            'std': np.std(image),
            'var': np.var(image),
            'min': np.min(image),
            'max': np.max(image),
            'median': np.median(image),
            'skewness': self._calculate_skewness(image),
            'kurtosis': self._calculate_kurtosis(image)
        }
        
        return features
    
    def extract_all_features(self, image):
        """
        Extract all features from image
        
        Args:
            image: Input grayscale image
            
        Returns:
            dict: Combined dictionary of all features
        """
        all_features = {}
        
        # Extract DWT features
        dwt_features = self.extract_dwt_features(image)
        all_features.update(dwt_features)
        
        # Extract GLCM features
        glcm_features = self.extract_glcm_features(image)
        all_features.update(glcm_features)
        
        # Extract HOG features
        hog_features, _ = self.extract_hog_features(image)
        all_features.update(hog_features)
        
        # Extract statistical features
        stat_features = self.extract_statistical_features(image)
        all_features.update(stat_features)
        
        return all_features
    
    @staticmethod
    def _calculate_entropy(image):
        """Calculate entropy of image"""
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    @staticmethod
    def _calculate_skewness(image):
        """Calculate skewness"""
        mean = np.mean(image)
        std = np.std(image)
        return np.mean(((image - mean) / std) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(image):
        """Calculate kurtosis"""
        mean = np.mean(image)
        std = np.std(image)
        return np.mean(((image - mean) / std) ** 4) - 3


def batch_extract_features(images, feature_extractor=None):
    """
    Extract features from multiple images
    
    Args:
        images: List or array of images
        feature_extractor: FeatureExtractor instance (creates new if None)
        
    Returns:
        list: List of feature dictionaries
    """
    if feature_extractor is None:
        feature_extractor = FeatureExtractor()
    
    features_list = []
    for img in images:
        features = feature_extractor.extract_all_features(img)
        features_list.append(features)
    
    return features_list
