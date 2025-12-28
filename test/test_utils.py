"""
Tests for emotion.utils module
"""
import unittest
import tempfile
import os
from emotion.utils.utils import get_size


class TestGetSize(unittest.TestCase):
    """Test get_size utility function"""

    def setUp(self):
        # Create temporary directory with test CSV files
        self.test_dir = tempfile.mkdtemp()

        # Create mock study directories
        for study_num in range(1, 8):
            study_dir = os.path.join(self.test_dir, f'study{study_num}')
            os.makedirs(study_dir, exist_ok=True)

            # Create a few CSV files
            for i in range(3):
                filepath = os.path.join(study_dir, f'study{study_num}_subject{i}.csv')
                with open(filepath, 'w') as f:
                    f.write('timestamp,ECG,EDA\n')
                    f.write(','.join(['1,2,3'] * 1000))

    def tearDown(self):
        # Clean up
        import shutil
        shutil.rmtree(self.test_dir)

    def test_get_size_returns_dict(self):
        """Test that get_size returns dictionary"""
        result = get_size(self.test_dir)
        self.assertIsInstance(result, dict)

    def test_get_size_has_all_studies(self):
        """Test that all study keys are present"""
        result = get_size(self.test_dir)
        for i in range(1, 8):
            self.assertIn(f'study{i}', result)

    def test_get_size_structure(self):
        """Test structure of returned data"""
        result = get_size(self.test_dir)
        for study_data in result.values():
            self.assertIn('Size', study_data)
            self.assertIn('File_Count', study_data)
            self.assertIsInstance(study_data['Size'], (int, float))
            self.assertIsInstance(study_data['File_Count'], int)

    def test_get_size_counts_files(self):
        """Test that files are counted correctly"""
        result = get_size(self.test_dir)
        # Each study should have 3 CSV files
        for i in range(1, 8):
            self.assertEqual(result[f'study{i}']['File_Count'], 3)


if __name__ == '__main__':
    unittest.main()
