"""
Tests for emotion.preprocessing.analysis module
"""
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests


class TestPlotPCA(unittest.TestCase):
    """Test plot_pca function"""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.style')
    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.color_palette')
    def test_plot_pca_runs(self, mock_palette, mock_figure, mock_style, mock_scatter,
                           mock_title, mock_xlabel, mock_ylabel, mock_legend,
                           mock_tight_layout, mock_savefig):
        """Test that plot_pca runs without error"""
        from emotion.preprocessing.analysis import plot_pca

        # Create sample data
        X = np.random.randn(100, 10)
        y = np.array(['happy'] * 50 + ['sad'] * 50)

        mock_palette.return_value = [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]

        plot_pca(X, y)

        mock_figure.assert_called_once()
        mock_savefig.assert_called_once_with("PCA_Study1.png")

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.style')
    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.color_palette')
    def test_plot_pca_with_multiple_emotions(self, mock_palette, mock_figure, mock_style,
                                              mock_scatter, mock_title, mock_xlabel, mock_ylabel,
                                              mock_legend, mock_tight_layout, mock_savefig):
        """Test plot_pca with multiple emotion labels"""
        from emotion.preprocessing.analysis import plot_pca

        X = np.random.randn(150, 5)
        y = np.array(['happy'] * 50 + ['sad'] * 50 + ['anger'] * 50)

        mock_palette.return_value = [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)]

        plot_pca(X, y)

        mock_scatter.assert_called()
class TestGetUniqueEmotionsForAllStudies(unittest.TestCase):
    """Test get_unique_emotions_for_all_studies function"""

    def test_get_unique_emotions(self):
        """Test collecting unique emotions from all studies"""
        from emotion.preprocessing.analysis import get_unique_emotions_for_all_studies

        mock_loader = Mock()
        mock_loader.get_unique_emotions.side_effect = [
            ['happy', 'sad'],
            ['happy', 'anger'],
            ['sad', 'fear'],
            ['joy', 'sadness'],
            ['happy'],
            ['anger', 'disgust'],
            ['fear', 'surprise']
        ]

        result = get_unique_emotions_for_all_studies(mock_loader)

        self.assertIsInstance(result, list)
        # Should have all unique emotions
        self.assertIn('happy', result)
        self.assertIn('sad', result)
        self.assertIn('anger', result)
        self.assertIn('fear', result)


class TestGenerateEmotionPresenceMatrix(unittest.TestCase):
    """Test generate_emotion_presence_matrix function"""

    def test_generate_matrix(self):
        """Test generating emotion presence matrix"""
        from emotion.preprocessing.analysis import generate_emotion_presence_matrix

        mock_loader = Mock()

        # Mock get_unique_emotions for each study
        mock_loader.get_unique_emotions.side_effect = [
            ['happy', 'sad'],
            ['happy'],
            ['sad'],
            ['happy', 'sad'],
            ['happy'],
            ['sad'],
            ['happy', 'sad']
        ]

        # Mock get_study_metadata
        def mock_metadata(study_num):
            df = pd.DataFrame({
                'EMOTION': ['happy'] * 5 + ['sad'] * 3,
                'FILE_NAME': [f'file_{i}' for i in range(8)]
            })
            return df

        mock_loader.get_study_metadata.side_effect = mock_metadata

        all_emotions = ['happy', 'sad']
        result = generate_emotion_presence_matrix(all_emotions, mock_loader)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 7)  # 7 studies
        self.assertIn('happy', result.columns)
        self.assertIn('sad', result.columns)


class TestGenerateHeatmap(unittest.TestCase):
    """Test generate_heatmap function"""

    @patch('emotion.preprocessing.analysis.plt')
    @patch('emotion.preprocessing.analysis.sns')
    def test_generate_heatmap_runs(self, mock_sns, mock_plt):
        """Test that generate_heatmap runs without error"""
        from emotion.preprocessing.analysis import generate_heatmap

        emotion_matrix = pd.DataFrame({
            'happy': [5, 3, 2, 4, 1, 6, 2],
            'sad': [3, 2, 4, 1, 5, 2, 3]
        }, index=['Study1', 'Study2', 'Study3', 'Study4', 'Study5', 'Study6', 'Study7'])

        generate_heatmap(emotion_matrix)

        mock_sns.heatmap.assert_called_once()
        mock_plt.show.assert_called_once()


class TestGenerateStudySizeBarChart(unittest.TestCase):
    """Test generate_study_size_bar_chart function"""

    @patch('emotion.preprocessing.analysis.utils')
    @patch('emotion.preprocessing.analysis.sns')
    def test_generate_bar_chart(self, mock_sns, mock_utils):
        """Test generating study size bar chart"""
        from emotion.preprocessing.analysis import generate_study_size_bar_chart

        mock_utils.get_size.return_value = {
            'study1': {'Size': 1.5, 'Files': 10},
            'study2': {'Size': 2.0, 'Files': 15},
        }

        mock_loader = Mock()
        mock_loader.data_dir = '/data/raw/'

        # Mock the Axes object
        mock_ax = Mock()
        mock_ax.containers = [Mock()]
        mock_sns.barplot.return_value = mock_ax

        generate_study_size_bar_chart(mock_loader)

        mock_utils.get_size.assert_called_once_with('/data/raw/')
        mock_sns.barplot.assert_called_once()


if __name__ == '__main__':
    unittest.main()
