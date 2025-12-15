"""
Tests for emotion.core.popane module - Main POPANE facade
"""
import unittest
from unittest.mock import Mock, MagicMock, patch
from emotion.core.popane import POPANE
from emotion.studies.subject import Subject, POPANEMetadata
from emotion.studies.study import Study


class TestPOPANE(unittest.TestCase):
    """Test POPANE facade class"""
    
    def setUp(self):
        """Set up test POPANE instance with mocked data loader"""
        with patch('emotion.core.popane.POPANEDataLoader') as mock_loader:
            self.mock_loader = mock_loader.return_value
            self.popane = POPANE(data_dir="data/raw/")
    
    # ==================== Initialization Tests ====================
    
    def test_initialization(self):
        """Test POPANE initializes with 7 studies"""
        self.assertIsNotNone(self.popane.loader)
        self.assertEqual(len(self.popane.studies), 7)
        for i in range(1, 8):
            self.assertIn(i, self.popane.studies)
            self.assertIsNotNone(self.popane.studies[i])
    
    def test_initialization_with_custom_data_dir(self):
        """Test POPANE initialization with custom data directory"""
        with patch('emotion.core.popane.POPANEDataLoader') as mock_loader:
            popane = POPANE(data_dir="custom/path/")
            mock_loader.assert_called_once_with(data_dir="custom/path/")
    
    # ==================== Study Access Tests ====================
    
    def test_get_study_valid(self):
        """Test getting a valid study"""
        study = self.popane.get_study(1)
        self.assertIsNotNone(study)
        
    def test_get_study_invalid(self):
        """Test getting an invalid study returns None"""
        study = self.popane.get_study(99)
        self.assertIsNone(study)
    
    def test_get_study_all_studies(self):
        """Test getting all 7 studies"""
        for study_num in range(1, 8):
            study = self.popane.get_study(study_num)
            self.assertIsNotNone(study)
    
    # ==================== Emotion Query Tests ====================
    
    def test_get_unique_emotions(self):
        """Test getting unique emotions for a study"""
        # Mock the study's get_unique_emotions method
        mock_study = self.popane.studies[1]
        mock_study.get_unique_emotions = Mock(return_value=['joy', 'anger', 'fear'])
        
        emotions = self.popane.get_unique_emotions(1)
        self.assertEqual(emotions, ['joy', 'anger', 'fear'])
        mock_study.get_unique_emotions.assert_called_once()
    
    def test_get_unique_emotions_invalid_study(self):
        """Test getting emotions for invalid study returns empty list"""
        emotions = self.popane.get_unique_emotions(99)
        self.assertEqual(emotions, [])
    
    def test_get_unique_emotions_for_all_studies(self):
        """Test getting all unique emotions across all studies"""
        # Mock each study to return different emotions
        for i in range(1, 8):
            mock_study = self.popane.studies[i]
            mock_study.get_unique_emotions = Mock(return_value=['joy', 'anger', f'emotion_{i}'])
        
        all_emotions = self.popane.get_unique_emotions_for_all_studies()
        
        # Should contain joy, anger, and emotion_1 through emotion_7
        self.assertIn('joy', all_emotions)
        self.assertIn('anger', all_emotions)
        for i in range(1, 8):
            self.assertIn(f'emotion_{i}', all_emotions)
        
        # Should be sorted
        self.assertEqual(all_emotions, sorted(all_emotions))
    
    # ==================== Subject Query Tests ====================
    
    def test_get_subject_ids(self):
        """Test getting subject IDs for a study"""
        mock_study = self.popane.studies[1]
        mock_study.get_subject_ids = Mock(return_value=[1, 2, 3, 4, 5])
        
        subject_ids = self.popane.get_subject_ids(1)
        self.assertEqual(subject_ids, [1, 2, 3, 4, 5])
        mock_study.get_subject_ids.assert_called_once()
    
    def test_get_subject_ids_invalid_study(self):
        """Test getting subject IDs for invalid study returns empty list"""
        subject_ids = self.popane.get_subject_ids(99)
        self.assertEqual(subject_ids, [])
    
    def test_get_subject(self):
        """Test getting a specific subject"""
        mock_subject = Mock(spec=Subject)
        mock_study = self.popane.studies[1]
        mock_study.get_subject = Mock(return_value=mock_subject)
        
        subject = self.popane.get_subject(1, 123)
        self.assertEqual(subject, mock_subject)
        mock_study.get_subject.assert_called_once_with(123, emotion=None)
    
    def test_get_subject_invalid_study(self):
        """Test getting subject from invalid study returns None"""
        subject = self.popane.get_subject(99, 123)
        self.assertIsNone(subject)
    
    # ==================== Metadata Query Tests ====================
    
    def test_get_study_metadata(self):
        """Test getting study metadata"""
        mock_metadata = Mock(spec=POPANEMetadata)
        mock_study = self.popane.studies[1]
        mock_study.get_study_metadata = Mock(return_value=mock_metadata)
        
        metadata = self.popane.get_study_metadata(1)
        self.assertEqual(metadata, mock_metadata)
        mock_study.get_study_metadata.assert_called_once()
    
    def test_get_study_metadata_invalid_study(self):
        """Test getting metadata for invalid study returns None"""
        metadata = self.popane.get_study_metadata(99)
        self.assertIsNone(metadata)
    
    def test_get_subject_metadata(self):
        """Test getting subject metadata"""
        mock_subject = Mock(spec=Subject)
        mock_study = self.popane.studies[1]
        mock_study.get_subject = Mock(return_value=mock_subject)
        
        subject = self.popane.get_subject_metadata(1, 123)
        self.assertEqual(subject, mock_subject)
        mock_study.get_subject.assert_called_once_with(123, emotion=None)
    
    # ==================== Data Retrieval Tests ====================
    
    def test_get_subjects_by_emotion(self):
        """Test getting subjects filtered by emotion"""
        mock_subjects = {1: Mock(spec=Subject), 2: Mock(spec=Subject)}
        self.mock_loader.get_all_subjects_from_study = Mock(return_value=mock_subjects)
        
        subjects = self.popane.get_subjects_by_emotion(1, ['joy', 'anger'])
        self.assertEqual(subjects, mock_subjects)
        self.mock_loader.get_all_subjects_from_study.assert_called_once_with(1, ['joy', 'anger'])
    
    def test_get_subjects_by_emotion_invalid_study(self):
        """Test getting subjects by emotion for invalid study returns empty dict"""
        subjects = self.popane.get_subjects_by_emotion(99, ['joy'])
        self.assertEqual(subjects, {})
    
    # ==================== Feature Analysis Tests ====================
    
    def test_get_features(self):
        """Test getting features for a study"""
        mock_study = self.popane.studies[1]
        mock_study.get_measurements = Mock(return_value={'ECG': 'float', 'EDA': 'float', 'TEMP': 'float'})
        
        features = self.popane.get_features(1)
        self.assertEqual(features, {'ECG', 'EDA', 'TEMP'})
    
    def test_get_features_invalid_study(self):
        """Test getting features for invalid study returns empty set"""
        features = self.popane.get_features(99)
        self.assertEqual(features, set())
    
    def test_get_features_no_measurements(self):
        """Test getting features when study has no measurements"""
        mock_study = self.popane.studies[1]
        mock_study.get_measurements = Mock(return_value=None)
        
        features = self.popane.get_features(1)
        self.assertEqual(features, set())
    
    def test_get_all_features(self):
        """Test getting all features across all studies"""
        # Mock different features for each study
        for i in range(1, 8):
            mock_study = self.popane.studies[i]
            measurements = {'ECG': 'float', 'EDA': 'float', f'FEATURE_{i}': 'float'}
            mock_study.get_measurements = Mock(return_value=measurements)
        
        all_features = self.popane.get_all_features()
        
        # Should contain common features and unique ones
        self.assertIn('ECG', all_features)
        self.assertIn('EDA', all_features)
        for i in range(1, 8):
            self.assertIn(f'FEATURE_{i}', all_features)
        
        # Should be sorted
        self.assertEqual(all_features, sorted(all_features))
    
    def test_get_common_features(self):
        """Test getting common features across specified studies"""
        # Study 1 and 2 have ECG, EDA
        # Study 3 has ECG, EDA, TEMP
        self.popane.studies[1].get_measurements = Mock(return_value={'ECG': 'float', 'EDA': 'float'})
        self.popane.studies[2].get_measurements = Mock(return_value={'ECG': 'float', 'EDA': 'float'})
        self.popane.studies[3].get_measurements = Mock(return_value={'ECG': 'float', 'EDA': 'float', 'TEMP': 'float'})
        
        common = self.popane.get_common_features([1, 2, 3])
        self.assertEqual(common, {'ECG', 'EDA'})
    
    def test_get_common_features_no_overlap(self):
        """Test getting common features when there's no overlap"""
        self.popane.studies[1].get_measurements = Mock(return_value={'ECG': 'float'})
        self.popane.studies[2].get_measurements = Mock(return_value={'EDA': 'float'})
        
        common = self.popane.get_common_features([1, 2])
        self.assertEqual(common, set())
    
    def test_get_common_features_empty_study_list(self):
        """Test getting common features with empty study list"""
        common = self.popane.get_common_features([])
        self.assertEqual(common, set())
    
    # ==================== Legacy API Tests ====================
    
    def test_get_data_for_subject_from_study_legacy(self):
        """Test legacy method get_data_for_subject_from_study"""
        mock_subject = Mock(spec=Subject)
        mock_study = self.popane.studies[1]
        mock_study.get_subject = Mock(return_value=mock_subject)
        
        subject = self.popane.get_data_for_subject_from_study(1, 123)
        self.assertEqual(subject, mock_subject)
    
    def test_get_data_from_study_with_emotions_legacy(self):
        """Test legacy method get_data_from_study_with_emotions"""
        mock_subjects = {1: Mock(spec=Subject)}
        self.mock_loader.get_all_subjects_from_study = Mock(return_value=mock_subjects)
        
        subjects = self.popane.get_data_from_study_with_emotions(1, ['joy'])
        self.assertEqual(subjects, mock_subjects)
    
    def test_get_features_for_study_legacy(self):
        """Test legacy method get_features_for_study"""
        mock_study = self.popane.studies[1]
        mock_study.get_measurements = Mock(return_value={'ECG': 'float'})
        
        features = self.popane.get_features_for_study(1)
        self.assertEqual(features, {'ECG'})
    
    def test_get_overlap_features_across_studies_legacy(self):
        """Test legacy method get_overlap_features_across_studies"""
        self.popane.studies[1].get_measurements = Mock(return_value={'ECG': 'float', 'EDA': 'float'})
        self.popane.studies[2].get_measurements = Mock(return_value={'ECG': 'float', 'EDA': 'float'})
        
        common = self.popane.get_overlap_features_across_studies([1, 2])
        self.assertEqual(common, {'ECG', 'EDA'})
    
    def test_get_subject_metadata_byID_legacy(self):
        """Test legacy method get_subject_metadata_byID"""
        mock_subject = Mock(spec=Subject)
        mock_study = self.popane.studies[1]
        mock_study.get_subject = Mock(return_value=mock_subject)
        
        subject = self.popane.get_subject_metadata_byID(1, 123)
        self.assertEqual(subject, mock_subject)


if __name__ == '__main__':
    unittest.main()
