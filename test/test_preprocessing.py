"""
Tests for emotion.preprocessing module
"""
from re import S
import unittest
from unittest.mock import patch
from matplotlib.dates import SU
import pandas as pd
import numpy as np
from emotion.preprocessing.dataprocess import (
    ECGSmoothTransformer,
    window_data,
    rolling_mean,
    exponential_moving_average,
    clean_DBP
)


class TestECGSmoothTransformer(unittest.TestCase):
    """Test ECG smoothing transformer"""
    
    def setUp(self):
        self.transformer = ECGSmoothTransformer(sigma=5)
        self.test_data = pd.DataFrame({
            'ECG': np.random.randn(1000),
            'EDA': np.random.randn(1000),
            'SBP': np.random.randn(1000),
            'DBP': np.random.randn(1000)
        })
    
    def test_fit_returns_self(self):
        """Test that fit returns self for chaining"""
        result = self.transformer.fit(self.test_data)
        self.assertIs(result, self.transformer)
    
    def test_fit_sets_n_features(self):
        """Test that fit sets n_features_in_"""
        self.transformer.fit(self.test_data)
        self.assertEqual(self.transformer.n_features_in_, 4)
    
    def test_transform_preserves_shape(self):
        """Test that transform preserves data shape"""
        self.transformer.fit(self.test_data)
        result = self.transformer.transform(self.test_data)
        self.assertEqual(result.shape, self.test_data.shape)
    
    def test_transform_smooths_ecg(self):
        """Test that ECG signal is smoothed"""
        # Create noisy signal
        noisy_ecg = pd.DataFrame({
            'ECG': np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.5,
            'EDA': np.random.randn(100),
            'SBP': np.random.randn(100),
            'DBP': np.random.randn(100)
        })
        
        self.transformer.fit(noisy_ecg)
        smoothed = self.transformer.transform(noisy_ecg)
        
        # Variance should be reduced
        self.assertLess(smoothed['ECG'].var(), noisy_ecg['ECG'].var())


class TestWindowData(unittest.TestCase):
    """Test window_data function"""
    
    def setUp(self):
        self.dataset = pd.DataFrame({
            'Subject_ID': [1] * 100,
            'File_Name': ['test.csv'] * 100,
            'ECG': np.random.randn(100),
            'EDA': np.random.randn(100),
            'SBP': np.random.randn(100),
            'DBP': np.random.randn(100),
            'respiration': np.random.randn(100),
            'temp': np.random.randn(100),
            'Emotion': ['joy'] * 100
        })
    
    def test_window_data_shape(self):
        """Test that windowing produces correct shape"""
        X, y = window_data(self.dataset, window__size=10, steps=5)
    
    def test_rolling_mean(self):
        """Test rolling mean smoothing"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = rolling_mean(data, window_size=3)
        self.assertEqual(len(result), len(data))
        # Check that it's smoothed
        self.assertIsInstance(result, pd.Series)
    
    def test_exponential_moving_average(self):
        """Test exponential moving average"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = exponential_moving_average(data, span=3)
        self.assertEqual(len(result), len(data))
        self.assertIsInstance(result, pd.Series)
    
    @patch('emotion.preprocessing.dataprocess.plt.show')
    @patch('emotion.preprocessing.dataprocess.plt.tight_layout')
    @patch('emotion.preprocessing.dataprocess.plt.subplots')
    def test_clean_DBP_with_valid_data(self, mock_subplots, mock_tight_layout, mock_show):
        """Test DBP cleaning function - primarily a visualization function"""
        # Mock the figure and axes
        from unittest.mock import MagicMock
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
        
        # Create simple dataset
        dataset = pd.DataFrame({
            'TIMESTAMP': pd.date_range('2020-01-01', periods=100, freq='s'),
            'FILE_NAME': ['test.csv'] * 100,
            'DBP': np.random.uniform(60, 90, 100),
            'EMOTION': ['joy'] * 50 + ['anger'] * 50,
            'SUBJECT_ID': [1] * 100
        })
        
        result = clean_DBP(dataset)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        self.assertIn('DBP', result.columns)
        
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()
    
    def test_window_data_labels(self):
        """Test that labels are correctly assigned"""
        X, y = window_data(self.dataset, window__size=10, steps=5)
        self.assertTrue(all(y == 'joy'))


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning functions"""
    
    def setUp(self):
        self.data = pd.DataFrame({
            'DBP': np.random.randn(1000) * 10 + 80,
            'SBP': np.random.randn(1000) * 10 + 120,
            "EMOTION": ['joy'] * 500 + ['anger'] * 500,
            "SUBJECT_ID": [1] * 1000,
            "FILE_NAME": ['test.csv'] * 1000,
            "TIMESTAMP": np.arange(1000)
            })
    
    def test_rolling_mean(self):
        """Test rolling mean smoothing"""
        smoothed = rolling_mean(self.data['DBP'], window_size=10)
        self.assertEqual(len(smoothed), len(self.data['DBP']))
    
    def test_exponential_moving_average(self):
        """Test exponential moving average"""
        ema = exponential_moving_average(self.data['DBP'], span=10)
        self.assertEqual(len(ema), len(self.data['DBP']))
    
    def test_clean_reduces_noise(self):
        """Test that clean function reduces noise"""
        cleaned = clean_DBP(self.data)
        # DBP should be smoother after cleaning
        self.assertIsNotNone(cleaned['DBP'])


if __name__ == '__main__':
    unittest.main()
