"""
Tests for emotion.studies module
"""
import unittest
from emotion.studies.study import Study, StudyConfig, Study1, Study2
from emotion.studies.subject import Subject
from emotion.dataloader.popaneloader import POPANEDataLoader, POPANEMETADataLoader
from emotion.studies.subject import POPANEMetadata
from studies.metadata import create_popane_metadata


class TestStudyConfig(unittest.TestCase):
    """Test StudyConfig dataclass"""
    
    def test_study_config_creation(self):
        """Test creating study configuration"""
        config = StudyConfig(
            number=1,
            name="test_study",
            measurements=("ECG", "EDA"),
            dtypes={"ECG": "float", "EDA": "float"}
        )
        self.assertEqual(config.number, 1)
        self.assertEqual(config.name, "test_study")
    
    def test_columns_property(self):
        """Test columns property returns set"""
        config = StudyConfig(
            number=1,
            name="test_study",
            measurements=("ECG", "EDA", "SBP"),
            dtypes={}
        )
        columns = config.columns
        self.assertIsInstance(columns, set)
        self.assertEqual(len(columns), 3)


class TestStudyClasses(unittest.TestCase):
    """Test Study class implementations"""
    
    def test_study1_config(self):
        """Test Study 1 configuration"""
        self.assertEqual(Study1.config.number, 1)
        self.assertEqual(Study1.config.name, "study1")
        self.assertIn("ECG", Study1.config.measurements)
        self.assertIn("EDA", Study1.config.measurements)
        self.assertIn("temp", Study1.config.measurements)
        self.assertIn("respiration", Study1.config.measurements)
    
    def test_study2_config(self):
        """Test Study 2 configuration"""
        self.assertEqual(Study2.config.number, 2)
        self.assertIn("CO", Study2.config.measurements)
        self.assertIn("TPR", Study2.config.measurements)
    
    def test_available_measurements(self):
        """Test available measurements property"""
        loader = POPANEDataLoader()
        study = Study1(loader)
        measurements = study.available_measurements
        self.assertIsInstance(measurements, tuple)
        self.assertGreater(len(measurements), 0)


class TestSubject(unittest.TestCase):
    """Test Subject class"""
    
    def setUp(self):
        import pandas as pd
        self.loader = POPANEDataLoader()
        
        # Create mock subject data
        self.test_data = pd.DataFrame({
            'timestamp': range(100),
            'ECG': range(100),
            'EDA': range(100),
            'Emotion': ['joy'] * 50 + ['anger'] * 50
        })
        
        # Mock study
        self.study = Study1(self.loader)
    
    def test_subject_repr(self):
        """Test subject string representation"""
        pmeta = POPANEMETADataLoader()
        meta_df = pmeta.get_subject_metadata(1, 1)
        meta_calss = create_popane_metadata(meta_df)
        subject = Subject(metadata=meta_calss);
        repr_str = repr(subject)
        self.assertIn('Subject', repr_str)
        self.assertIn("Subject(study=['STUDY1', 'STUDY1', 'STUDY1']", repr_str)
        self.assertIn('id=[1, 1, 1]', repr_str)


if __name__ == '__main__':
    unittest.main()
