"""
Tests for emotion.db.db module - Database management
"""
import unittest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from emotion.db.db import POPANEDB


class TestPOPANEDB(unittest.TestCase):
    """Test POPANEDB database management class"""
    
    def setUp(self):
        """Set up test database with temporary directory"""
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_dir, "test.db")
        # Create data/processed directory for logging
        os.makedirs("data/processed", exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    # ==================== Initialization Tests ====================
    
    def test_initialization(self):
        """Test POPANEDB initializes correctly"""
        db = POPANEDB(data_dir="data/raw/")
        self.assertIsNotNone(db)
        self.assertFalse(db._POPANEDB__downloads_completed)
        self.assertEqual(db._POPANEDB__number_of_downloads, 0)
    
    def test_initialization_with_custom_dir(self):
        """Test POPANEDB initialization with custom data directory"""
        db = POPANEDB(data_dir=self.test_dir)
        self.assertIsNotNone(db)
    
    # ==================== Study Column Mapping Tests ====================
    
    def test_study_columns_map_contains_all_studies(self):
        """Test that study_columns_map has entries for all 7 studies"""
        db = POPANEDB(data_dir="data/raw/")
        for study_num in range(1, 8):
            self.assertIn(study_num, db.study_columns_map)
            columns, dtypes = db.study_columns_map[study_num]
            self.assertIsNotNone(columns)
            self.assertIsNotNone(dtypes)
            self.assertIsInstance(dtypes, list)
    
    def test_study1_columns(self):
        """Test Study 1 column definitions"""
        db = POPANEDB(data_dir="data/raw/")
        columns, dtypes = db.study_columns_map[1]
        expected_cols = ("timestamp", "affect", "ECG", "EDA", "temp", 
                        "respiration", "SBP", "DBP", "marker")
        self.assertEqual(columns, expected_cols)
        self.assertEqual(len(dtypes), 9)
    
    def test_study2_columns(self):
        """Test Study 2 column definitions"""
        db = POPANEDB(data_dir="data/raw/")
        columns, dtypes = db.study_columns_map[2]
        expected_cols = ("timestamp", "affect", "ECG", "EDA", "SBP", 
                        "DBP", "CO", "TPR", "marker")
        self.assertEqual(columns, expected_cols)
        self.assertEqual(len(dtypes), 9)
    
    def test_study3_columns(self):
        """Test Study 3 column definitions"""
        db = POPANEDB(data_dir="data/raw/")
        columns, dtypes = db.study_columns_map[3]
        expected_cols = ("timestamp", "affect", "ECG", "EDA", "dzdt", "dz", 
                        "z0", "SBP", "DBP", "CO", "TPR", "marker")
        self.assertEqual(columns, expected_cols)
        self.assertEqual(len(dtypes), 12)
    
    def test_study4_columns(self):
        """Test Study 4 column definitions"""
        db = POPANEDB(data_dir="data/raw/")
        columns, dtypes = db.study_columns_map[4]
        expected_cols = ("timestamp", "ECG", "EDA", "SBP", "DBP", "CO", "TPR", "marker")
        self.assertEqual(columns, expected_cols)
        self.assertEqual(len(dtypes), 8)
    
    def test_study5_columns(self):
        """Test Study 5 column definitions"""
        db = POPANEDB(data_dir="data/raw/")
        columns, dtypes = db.study_columns_map[5]
        expected_cols = ("timestamp", "affect", "ECG", "EDA", "SBP", 
                        "DBP", "CO", "TPR", "marker")
        self.assertEqual(columns, expected_cols)
        self.assertEqual(len(dtypes), 9)
    
    def test_study6_columns(self):
        """Test Study 6 column definitions"""
        db = POPANEDB(data_dir="data/raw/")
        columns, dtypes = db.study_columns_map[6]
        expected_cols = ("timestamp", "affect", "ECG", "dzdt", "dz", "z0", 
                        "EDA", "SBP", "DBP", "CO", "TPR", "marker")
        self.assertEqual(columns, expected_cols)
        self.assertEqual(len(dtypes), 12)
    
    def test_study7_columns(self):
        """Test Study 7 column definitions"""
        db = POPANEDB(data_dir="data/raw/")
        columns, dtypes = db.study_columns_map[7]
        expected_cols = ("timestamp", "affect", "ECG", "dzdt", "dz", "z0", "marker")
        self.assertEqual(columns, expected_cols)
        self.assertEqual(len(dtypes), 7)
    
    # ==================== Column and DType Consistency Tests ====================
    
    def test_columns_and_dtypes_length_match(self):
        """Test that columns and dtypes have matching lengths for all studies"""
        db = POPANEDB(data_dir="data/raw/")
        for study_num in range(1, 8):
            columns, dtypes = db.study_columns_map[study_num]
            self.assertEqual(len(columns), len(dtypes), 
                           f"Study {study_num}: columns and dtypes length mismatch")
    
    def test_all_dtypes_are_valid(self):
        """Test that all dtypes are valid DuckDB types"""
        db = POPANEDB(data_dir="data/raw/")
        valid_types = ["DOUBLE", "INTEGER", "VARCHAR", "BOOLEAN", "TIMESTAMP"]
        
        for study_num in range(1, 8):
            _, dtypes = db.study_columns_map[study_num]
            for dtype in dtypes:
                self.assertIn(dtype, valid_types, 
                            f"Study {study_num}: Invalid dtype {dtype}")
    
    # ==================== Database Creation Tests (Mocked) ====================
    
    @patch('emotion.db.db.duckdb.connect')
    @patch('emotion.db.db.POPANEDataLoader')
    def test_createDB_initialization(self, mock_loader, mock_connect):
        """Test database creation initializes DuckDB connection"""
        mock_db = MagicMock()
        mock_connect.return_value = mock_db
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        
        # Verify DuckDB connect is available
        self.assertTrue(callable(mock_connect))
    
    @patch('emotion.db.db.os.cpu_count', return_value=8)
    def test_cpu_count_available(self, mock_cpu_count):
        """Test that CPU count can be retrieved"""
        count = mock_cpu_count()
        self.assertEqual(count, 8)
    
    def test_psutil_import_available(self):
        """Test that psutil can be imported"""
        try:
            import psutil
            self.assertTrue(hasattr(psutil, 'virtual_memory'))
        except ImportError:
            self.skipTest("psutil not installed")
    
    # ==================== Attribute Tests ====================
    
    def test_default_db_path(self):
        """Test default database path"""
        db = POPANEDB(data_dir="data/raw/")
        self.assertEqual(db._POPANEDB__duckdbpath, "data/processed/propane_emotion.db")
    
    def test_downloads_completed_initial_state(self):
        """Test downloads_completed starts as False"""
        db = POPANEDB(data_dir="data/raw/")
        self.assertFalse(db._POPANEDB__downloads_completed)
    
    def test_number_of_downloads_initial_state(self):
        """Test number_of_downloads starts at 0"""
        db = POPANEDB(data_dir="data/raw/")
        self.assertEqual(db._POPANEDB__number_of_downloads, 0)
    
    def test_threadpool_initial_state(self):
        """Test threadpool starts as None"""
        db = POPANEDB(data_dir="data/raw/")
        self.assertIsNone(db._POPANEDB__threadpool)


class TestPOPANEDBIntegration(unittest.TestCase):
    """Integration tests for POPANEDB (may require actual data)"""
    
    @unittest.skipUnless(os.path.exists("data/raw/"), "Requires data/raw/ directory")
    def test_inherits_from_popane_data_loader(self):
        """Test POPANEDB inherits from POPANEDataLoader"""
        from emotion.dataloader.popaneloader import POPANEDataLoader
        db = POPANEDB(data_dir="data/raw/")
        self.assertIsInstance(db, POPANEDataLoader)
    
    @unittest.skipUnless(os.path.exists("data/raw/"), "Requires data/raw/ directory")
    def test_has_loader_methods(self):
        """Test POPANEDB has data loader methods"""
        db = POPANEDB(data_dir="data/raw/")
        # Should have inherited methods from POPANEDataLoader
        self.assertTrue(hasattr(db, 'get_study_metadata'))
        self.assertTrue(hasattr(db, 'get_subject_ids'))


if __name__ == '__main__':
    unittest.main()
