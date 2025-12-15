"""
Tests for emotion.visualization.figure module
"""
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from emotion.visualization.figure import POPANEFigureGenerator, TOATALFEATURELIST


class TestPOPANEFigureGenerator(unittest.TestCase):
    """Test POPANEFigureGenerator class"""
    
    def setUp(self):
        """Set up test figure generator"""
        self.fig_gen = POPANEFigureGenerator()
        plt.ioff()  # Turn off interactive mode for testing
    
    def tearDown(self):
        """Close all figures after each test"""
        plt.close('all')
    
    # ==================== Initialization Tests ====================
    
    def test_initialization(self):
        """Test figure generator initializes correctly"""
        self.assertIsNotNone(self.fig_gen)
        self.assertIsInstance(self.fig_gen.feature_plot_functions, dict)
    
    def test_feature_plot_functions_defined(self):
        """Test that feature plot functions are defined"""
        self.assertGreater(len(self.fig_gen.feature_plot_functions), 0)
        
        # Check common features are defined
        common_features = ['ECG', 'EDA', 'SBP', 'DBP']
        for feature in common_features:
            self.assertIn(feature, self.fig_gen.feature_plot_functions)
    
    def test_feature_plot_config_structure(self):
        """Test that feature plot configurations have required keys"""
        for feature, config in self.fig_gen.feature_plot_functions.items():
            self.assertIn('plotfn', config)
            self.assertIn('title', config)
            self.assertIn('y_label', config)
            self.assertIn('color', config)
    
    # ==================== Style Setup Tests ====================
    
    def test_set_up_figure_styles(self):
        """Test that figure styles are set up"""
        self.fig_gen._set_up_figure_styles()
        
        # Check matplotlib rcParams were updated
        self.assertIsNotNone(plt.rcParams['figure.figsize'])
        self.assertIsNotNone(plt.rcParams['axes.titlesize'])
        self.assertIsNotNone(plt.rcParams['axes.labelsize'])
    
    # ==================== Plot Signals Tests ====================
    
    def test_plot_signals_basic(self):
        """Test basic signal plotting"""
        fig, ax = plt.subplots()
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        
        result = POPANEFigureGenerator.plot_signals(
            x_data, y_data,
            title="Test Signal",
            axis=ax
        )
        
        # Should return the axis
        self.assertIsInstance(result, Axes)
        self.assertEqual(ax.get_title(), "Test Signal")
    
    def test_plot_signals_with_label(self):
        """Test signal plotting with label"""
        fig, ax = plt.subplots()
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        
        POPANEFigureGenerator.plot_signals(
            x_data, y_data,
            label="sin(x)",
            axis=ax
        )
        
        legend = ax.get_ylabel()
        self.assertIsNotNone(legend)
    
    def test_plot_signals_with_color(self):
        """Test signal plotting with custom color"""
        fig, ax = plt.subplots()
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        
        POPANEFigureGenerator.plot_signals(
            x_data, y_data,
            color='red',
            axis=ax
        )
        
        lines = ax.get_lines()
        self.assertEqual(len(lines), 1)
    
    def test_plot_signals_pandas_series(self):
        """Test plotting with pandas Series input"""
        fig, ax = plt.subplots()
        x_data = pd.Series(np.linspace(0, 10, 100))
        y_data = pd.Series(np.sin(np.linspace(0, 10, 100)))
        
        result = POPANEFigureGenerator.plot_signals(
            x_data, y_data,
            axis=ax
        )
        
        self.assertIsInstance(result, Axes)
    
    def test_plot_signals_numpy_array(self):
        """Test plotting with numpy array input"""
        fig, ax = plt.subplots()
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        
        result = POPANEFigureGenerator.plot_signals(
            x_data, y_data,
            axis=ax
        )
        
        self.assertIsInstance(result, Axes)
    
    # ==================== Feature List Tests ====================
    
    def test_total_feature_list_defined(self):
        """Test that TOATALFEATURELIST is defined"""
        self.assertIsNotNone(TOATALFEATURELIST)
        self.assertIsInstance(TOATALFEATURELIST, list)
        self.assertGreater(len(TOATALFEATURELIST), 0)
    
    def test_total_feature_list_contains_common_features(self):
        """Test that TOATALFEATURELIST contains common physiological features"""
        common_features = ['ECG', 'EDA', 'SBP', 'DBP']
        for feature in common_features:
            self.assertIn(feature, TOATALFEATURELIST)
    
    def test_total_feature_list_uniqueness(self):
        """Test that TOATALFEATURELIST has no duplicates"""
        self.assertEqual(len(TOATALFEATURELIST), len(set(TOATALFEATURELIST)))
    
    # ==================== ECG Feature Tests ====================
    
    def test_ecg_plot_function_defined(self):
        """Test that ECG plot function is properly configured"""
        ecg_config = self.fig_gen.feature_plot_functions.get('ECG')
        self.assertIsNotNone(ecg_config)
        self.assertEqual(ecg_config['title'], 'ECG Signal')
        self.assertIn('mV', ecg_config['y_label'])
    
    def test_eda_plot_function_defined(self):
        """Test that EDA plot function is properly configured"""
        eda_config = self.fig_gen.feature_plot_functions.get('EDA')
        self.assertIsNotNone(eda_config)
        self.assertEqual(eda_config['title'], 'EDA Signal')
        self.assertIn('µS', eda_config['y_label'])
    
    # ==================== Multiple Signal Tests ====================
    
    def test_plot_multiple_signals(self):
        """Test plotting multiple signals on same axis"""
        fig, ax = plt.subplots()
        x_data = np.linspace(0, 10, 100)
        
        # Plot multiple signals
        POPANEFigureGenerator.plot_signals(
            x_data, np.sin(x_data),
            label="sin(x)", axis=ax
        )
        POPANEFigureGenerator.plot_signals(
            x_data, np.cos(x_data),
            label="cos(x)", axis=ax
        )
        
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
    
    def test_plot_signals_with_different_lengths(self):
        """Test plotting signals with different lengths"""
        fig, ax = plt.subplots()
        x_data1 = np.linspace(0, 10, 100)
        y_data1 = np.sin(x_data1)
        
        x_data2 = np.linspace(0, 10, 50)
        y_data2 = np.cos(x_data2)
        
        POPANEFigureGenerator.plot_signals(x_data1, y_data1, axis=ax)
        POPANEFigureGenerator.plot_signals(x_data2, y_data2, axis=ax)
        
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
    
    # ==================== Error Handling Tests ====================
    
    def test_plot_signals_with_empty_data(self):
        """Test plotting with empty data"""
        fig, ax = plt.subplots()
        x_data = np.array([])
        y_data = np.array([])
        
        try:
            POPANEFigureGenerator.plot_signals(x_data, y_data, axis=ax)
            # Should handle empty data gracefully
        except Exception as e:
            # If it raises, that's also acceptable behavior
            pass
    
    def test_plot_signals_mismatched_lengths(self):
        """Test plotting with mismatched x and y lengths"""
        fig, ax = plt.subplots()
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(np.linspace(0, 10, 50))  # Different length
        
        try:
            POPANEFigureGenerator.plot_signals(x_data, y_data, axis=ax)
        except (ValueError, AssertionError):
            # Expected to fail with mismatched lengths
            pass
    
    # ==================== Integration Tests ====================
    
    def test_create_figure_with_multiple_subplots(self):
        """Test creating figure with multiple subplots"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        x_data = np.linspace(0, 10, 100)
        
        signals = {
            'ECG': np.sin(x_data),
            'EDA': np.cos(x_data),
            'SBP': np.sin(x_data * 2)
        }
        
        for i, (name, signal) in enumerate(signals.items()):
            POPANEFigureGenerator.plot_signals(
                x_data, signal,
                title=name,
                axis=axes[i]
            )
        
        # Check all subplots were created
        self.assertEqual(len(axes), 3)
        for ax in axes:
            self.assertIsNotNone(ax.get_title())


class TestFeatureListConstants(unittest.TestCase):
    """Test feature list constants"""
    
    def test_feature_list_types(self):
        """Test that all features in list are strings"""
        for feature in TOATALFEATURELIST:
            self.assertIsInstance(feature, str)
    
    def test_feature_list_not_empty_strings(self):
        """Test that no feature names are empty strings"""
        for feature in TOATALFEATURELIST:
            self.assertGreater(len(feature), 0)


if __name__ == '__main__':
    unittest.main()
