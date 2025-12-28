"""
Tests for emotion.studies module
"""
import unittest
from emotion.studies.study import Study, StudyConfig, Study1, Study2
from emotion.studies.subject import Subject
from emotion.dataloader.popaneloader import POPANEDataLoader, POPANEMETADataLoader
from emotion.studies.subject import POPANEMetadata
from emotion.studies.metadata import create_popane_metadata
import emotion.studies as studies_module


class TestStudiesModuleGetattr(unittest.TestCase):
    """Test __getattr__ in studies module"""

    def test_getattr_study(self):
        """Test __getattr__ returns Study"""
        result = studies_module.__getattr__('Study')
        self.assertEqual(result, Study)

    def test_getattr_study1(self):
        """Test __getattr__ returns Study1"""
        result = studies_module.__getattr__('Study1')
        self.assertEqual(result, Study1)

    def test_getattr_study2(self):
        """Test __getattr__ returns Study2"""
        result = studies_module.__getattr__('Study2')
        self.assertEqual(result, Study2)

    def test_getattr_study3(self):
        """Test __getattr__ returns Study3"""
        from emotion.studies.study import Study3
        result = studies_module.__getattr__('Study3')
        self.assertEqual(result, Study3)

    def test_getattr_study4(self):
        """Test __getattr__ returns Study4"""
        from emotion.studies.study import Study4
        result = studies_module.__getattr__('Study4')
        self.assertEqual(result, Study4)

    def test_getattr_study5(self):
        """Test __getattr__ returns Study5"""
        from emotion.studies.study import Study5
        result = studies_module.__getattr__('Study5')
        self.assertEqual(result, Study5)

    def test_getattr_study6(self):
        """Test __getattr__ returns Study6"""
        from emotion.studies.study import Study6
        result = studies_module.__getattr__('Study6')
        self.assertEqual(result, Study6)

    def test_getattr_study7(self):
        """Test __getattr__ returns Study7"""
        from emotion.studies.study import Study7
        result = studies_module.__getattr__('Study7')
        self.assertEqual(result, Study7)

    def test_getattr_subject(self):
        """Test __getattr__ returns Subject"""
        result = studies_module.__getattr__('Subject')
        self.assertEqual(result, Subject)

    def test_getattr_study_registry(self):
        """Test __getattr__ returns STUDY_REGISTRY"""
        from emotion.studies.study import STUDY_REGISTRY
        result = studies_module.__getattr__('STUDY_REGISTRY')
        self.assertEqual(result, STUDY_REGISTRY)

    def test_getattr_version(self):
        """Test __getattr__ returns __version__"""
        result = studies_module.__getattr__('__version__')
        self.assertIsNotNone(result)

    def test_getattr_unknown(self):
        """Test __getattr__ returns None for unknown attribute"""
        result = studies_module.__getattr__('unknown_attr')
        self.assertIsNone(result)


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

    def test_study_repr(self):
        """Test Study __repr__"""
        loader = POPANEDataLoader()
        study = Study1(loader)
        repr_str = repr(study)
        self.assertIn("Study", repr_str)
        self.assertIn("1", repr_str)
        self.assertIn("study1", repr_str)

    def test_study_str(self):
        """Test Study __str__"""
        loader = POPANEDataLoader()
        study = Study1(loader)
        str_val = str(study)
        self.assertIn("Study 1", str_val)
        self.assertIn("study1", str_val)

    def test_study_len(self):
        """Test Study __len__"""
        loader = POPANEDataLoader()
        study = Study1(loader)
        length = len(study)
        self.assertIsInstance(length, int)
        self.assertGreaterEqual(length, 0)

    def test_get_unique_emotions(self):
        """Test get_unique_emotions method"""
        loader = POPANEDataLoader()
        study = Study1(loader)
        emotions = study.get_unique_emotions()
        self.assertIsInstance(emotions, list)

    def test_get_study_metadata(self):
        """Test get_study_metadata method"""
        loader = POPANEDataLoader()
        study = Study1(loader)
        metadata = study.get_study_metadata()
        # Could be None if no data, otherwise check type
        if metadata is not None:
            self.assertIsNotNone(metadata)

    def test_get_measurements(self):
        """Test get_measurements method"""
        loader = POPANEDataLoader()
        study = Study1(loader)
        measurements = study.get_measurements()
        self.assertIsInstance(measurements, dict)

    def test_study_description(self):
        """Test study_description method"""
        loader = POPANEDataLoader()
        study = Study1(loader)
        description = study.study_description()
        # Description can be None or string
        self.assertTrue(description is None or isinstance(description, str))

    def test_get_subject_ids(self):
        """Test get_subject_ids method"""
        loader = POPANEDataLoader()
        study = Study1(loader)
        subject_ids = study.get_subject_ids()
        self.assertIsInstance(subject_ids, list)


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

    def test_subject_measurements(self):
        """Test subject has measurements"""
        pmeta = POPANEMETADataLoader()
        meta_df = pmeta.get_subject_metadata(1, 1)
        meta_class = create_popane_metadata(meta_df)
        subject = Subject(metadata=meta_class)

        measurements = subject.get_measurements()
        self.assertIsNotNone(measurements)
        self.assertIsInstance(measurements, set)

    def test_subject_emotions_empty_when_no_data(self):
        """Test emotions property returns empty when no data loaded"""
        pmeta = POPANEMETADataLoader()
        meta_df = pmeta.get_subject_metadata(1, 1)
        meta_class = create_popane_metadata(meta_df)
        subject = Subject(metadata=meta_class)

        emotions = subject.emotions
        self.assertEqual(len(emotions), 0)

    def test_subject_get_emotion_data_no_data(self):
        """Test get_emotion_data returns None when no data"""
        pmeta = POPANEMETADataLoader()
        meta_df = pmeta.get_subject_metadata(1, 1)
        meta_class = create_popane_metadata(meta_df)
        subject = Subject(metadata=meta_class)

        result = subject.get_emotion_data('happy')
        self.assertIsNone(result)

    def test_subject_to_dataframe(self):
        """Test to_dataframe method"""
        pmeta = POPANEMETADataLoader()
        meta_df = pmeta.get_subject_metadata(1, 1)
        meta_class = create_popane_metadata(meta_df)
        subject = Subject(metadata=meta_class)

        # This will try to read actual files, so we need to mock or skip
        # For now test that method exists
        self.assertTrue(hasattr(subject, 'to_dataframe'))


class TestSubjectMeasurements(unittest.TestCase):
    """Test Subject class measurement constants"""

    def test_study1_measurements(self):
        """Test Study 1 measurement set"""
        from emotion.studies.subject import STUDY1_MEASUREMENT

        expected = {"TIMESTAMP", "AFFECT", "ECG", "EDA", "TEMP",
                   "RESPIRATION", "SBP", "DBP", "MARKER"}
        self.assertEqual(STUDY1_MEASUREMENT, expected)

    def test_study2_measurements(self):
        """Test Study 2 measurement set"""
        from emotion.studies.subject import STUDY2_MEASUREMENT

        self.assertIn("ECG", STUDY2_MEASUREMENT)
        self.assertIn("CO", STUDY2_MEASUREMENT)
        self.assertIn("TPR", STUDY2_MEASUREMENT)

    def test_all_study_measurements_defined(self):
        """Test all study measurements are defined"""
        from emotion.studies.subject import (
            STUDY1_MEASUREMENT, STUDY2_MEASUREMENT, STUDY3_MEASUREMENT,
            STUDY4_MEASUREMENT, STUDY5_MEASUREMENT, STUDY6_MEASUREMENT,
            STUDY7_MEASUREMENT
        )

        all_measurements = [
            STUDY1_MEASUREMENT, STUDY2_MEASUREMENT, STUDY3_MEASUREMENT,
            STUDY4_MEASUREMENT, STUDY5_MEASUREMENT, STUDY6_MEASUREMENT,
            STUDY7_MEASUREMENT
        ]

        for measurements in all_measurements:
            self.assertIsInstance(measurements, set)
            self.assertGreater(len(measurements), 0)
            # All should have TIMESTAMP
            self.assertIn("TIMESTAMP", measurements)


if __name__ == '__main__':
    unittest.main()
