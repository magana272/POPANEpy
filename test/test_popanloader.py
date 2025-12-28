"""
Comprehensive tests for POPANE emotion analysis package
"""
import os.path
import unittest
import dataclasses
import pandas as pd
import numpy as np

import emotion.dataloader.popaneloader as popane_data_loader
import emotion.dataloader as dataloader_module
from emotion.studies.subject import Subject
from emotion.preprocessing.dataprocess import ECGSmoothTransformer, window_data
from emotion.visualization.figure import POPANEFigureGenerator
from emotion.models.random_forest import EmotionRandomForest
from emotion import POPANE
from emotion.dataloader.downloader import POPANEDownloader


class TestDataloaderModuleGetattr(unittest.TestCase):
    """Test __getattr__ in dataloader module"""

    def test_getattr_popane_data_loader(self):
        """Test __getattr__ returns PopaneDataLoader"""
        result = dataloader_module.__getattr__('PopaneDataLoader')
        self.assertEqual(result, popane_data_loader.POPANEDataLoader)

    def test_getattr_popane_downloader(self):
        """Test __getattr__ returns POPANEDownloader"""
        result = dataloader_module.__getattr__('POPANEDownloader')
        self.assertEqual(result, POPANEDownloader)

    def test_getattr_version(self):
        """Test __getattr__ returns __version__"""
        result = dataloader_module.__getattr__('__version__')
        self.assertIsNotNone(result)

    def test_getattr_unknown(self):
        """Test __getattr__ returns None for unknown attribute"""
        result = dataloader_module.__getattr__('unknown_attr')
        self.assertIsNone(result)


class TestPOPANEMETADATALoader(unittest.TestCase):
    def setUp(self):
        self.loader = popane_data_loader.POPANEMETADataLoader()

    def test_get_study_metadata(self):
        study1_meta = self.loader.get_study_metadata(1)
        study2_meta = self.loader.get_study_metadata(2)
        study3_meta = self.loader.get_study_metadata(3)
        study4_meta = self.loader.get_study_metadata(4)
        study5_meta = self.loader.get_study_metadata(5)
        study6_meta = self.loader.get_study_metadata(6)
        study7_meta = self.loader.get_study_metadata(7)
        self.assertIsNotNone(study1_meta)
        self.assertIsNotNone(study2_meta)
        self.assertIsNotNone(study3_meta)
        self.assertIsNotNone(study4_meta)
        self.assertIsNotNone(study5_meta)
        self.assertIsNotNone(study6_meta)
        self.assertIsNotNone(study7_meta)

class test_POPANEDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = popane_data_loader.POPANEDataLoader(data_dir="data/raw/")

    def test_get_study_metadata(self):
        study1_meta = self.loader.get_study_metadata(1)
        study2_meta = self.loader.get_study_metadata(2)
        study3_meta = self.loader.get_study_metadata(3)
        study4_meta = self.loader.get_study_metadata(4)
        study5_meta = self.loader.get_study_metadata(5)
        study6_meta = self.loader.get_study_metadata(6)
        study7_meta = self.loader.get_study_metadata(7)
        self.assertIsNotNone(study1_meta)
        self.assertIsNotNone(study2_meta)

    def test_study2_metadata(self):
        study2_meta = self.loader.get_study_metadata(2)
        if study2_meta is not None:
            study2_cols = ["SUBJECT_ID", "SEX", "AGE", "HEIGHT", "WEIGHT", "STIMULI1", "STIMULI2",
                                               "STUDY_NAME", "FILE_NAME", "EMOTION", "FILE_PATH"]
            for col in study2_cols:
                self.assertIn(col, study2_meta.columns)
    def test_study3_metadata(self):
        study3_meta = self.loader.get_study_metadata(3)
        if study3_meta is not None:
            study3_cols = ["SUBJECT_ID", "SEX", "AGE", "HEIGHT", "WEIGHT",
                                               "STIMULI1", "STIMULI2", "STUDY_NAME", "FILE_NAME", "EMOTION", "FILE_PATH"]
            for col in study3_cols:
                self.assertIn(col, study3_meta.columns) # type: ignore
            self.assertEqual(len(study3_meta.columns), 11) # type: ignore

    def test_study4_metadata(self):
        study4_meta = self.loader.get_study_metadata(4)
        if study4_meta is not None:
            study4_cols = ["SUBJECT_ID", "SEX", "AGE", "HEIGHT", "WEIGHT", "STIMULI",
                                               "STUDY_NAME", "FILE_NAME", "EMOTION", "FILE_PATH"]
            for col in study4_cols:
                self.assertIn(col, study4_meta.columns) # type: ignore
            self.assertEqual(len(study4_meta.columns), 10) # type: ignore

    def test_study5_metadata(self):
        study5_meta = self.loader.get_study_metadata(5)
        if study5_meta is not None:
            study5_cols =   ["SUBJECT_ID", "SEX", "AGE", "HEIGHT", "WEIGHT",
                                               "STIMULI1", "STIMULI2", "STIMULI3", "STUDY_NAME",
                                               "FILE_NAME", "EMOTION", "FILE_PATH"]
            for col in study5_cols:
                self.assertIn(col, study5_meta.columns) # type: ignore
            self.assertEqual(len(study5_meta.columns), 12) # type: ignore

    def test_study6_metadata(self):
        study6_meta = self.loader.get_study_metadata(6)
        if study6_meta is not None:
            study6_cols = ["SUBJECT_ID", "SEX", "AGE", "HEIGHT", "WEIGHT", "STIMULI1", "STIMULI2",
                                               "STIMULI3", "STIMULI4", "STIMULI5", "STIMULI6",
                                               "STUDY_NAME", "FILE_NAME", "EMOTION", "FILE_PATH"]
            for col in study6_cols:
                self.assertIn(col, study6_meta.columns) # pyright: ignore[reportArgumentType]
            self.assertEqual(len(study6_meta.columns), 15) # type: ignore

    def test_study7_metadata(self):
        study7_meta = self.loader.get_study_metadata(7)
        if study7_meta is not None:
            study7_cols = ["SUBJECT_ID", "SEX", "AGE", "HEIGHT", "WEIGHT", "STIMULI1", "STIMULI2",
                                               "STIMULI3", "STIMULI4", "STIMULI5",
                                               "STUDY_NAME", "FILE_NAME", "EMOTION", "FILE_PATH"]
            for col in study7_cols:
                self.assertIn(col, study7_meta.columns) # type: ignore
            self.assertEqual(len(study7_meta.columns), 14) # pyright: ignore[reportArgumentType]

    def test_get_data_for_subject_from_study(self):
        subject_data: pd.DataFrame = self.loader.get_data_for_subject_from_study(1, 1)
        if subject_data is not None:
            self.assertIn('timestamp', subject_data.columns)
            self.assertIn('affect', subject_data.columns)
            self.assertIn('ECG', subject_data.columns)
            self.assertIn('EDA', subject_data.columns)
            self.assertIn('temp', subject_data.columns)
            self.assertIn('respiration', subject_data.columns)
            self.assertIn('SBP', subject_data.columns)
            self.assertIn('DBP', subject_data.columns)
            self.assertIn('marker', subject_data.columns)
        self.assertIsNotNone(subject_data)


class TestPOPANEMETADataLoaderAdditional(unittest.TestCase):
    """Additional tests for POPANEMETADataLoader"""

    def setUp(self):
        self.loader = popane_data_loader.POPANEMETADataLoader()

    def test_get_stimuli(self):
        """Test getting stimuli metadata"""
        stimuli = self.loader.get_stimuli()
        self.assertIsNotNone(stimuli)
        if stimuli is not None:
            self.assertIsInstance(stimuli, pd.DataFrame)

    def test_get_subject_metadata(self):
        """Test getting subject metadata"""
        subject_meta = self.loader.get_subject_metadata(1, 1)
        self.assertIsNotNone(subject_meta)
        if subject_meta is not None:
            self.assertIsInstance(subject_meta, pd.DataFrame)

    def test_get_all_studies_metadata(self):
        """Test getting all studies metadata"""
        all_meta = self.loader.get_all_studies_metadata()
        self.assertIsInstance(all_meta, dict)
        self.assertEqual(len(all_meta), 7)

    def test_is_cached(self):
        """Test is_cached method"""
        result = self.loader.is_cached()
        self.assertIsInstance(result, bool)

    def test_get_study_metadata_with_emotions_filter(self):
        """Test getting study metadata filtered by emotions"""
        # First get unfiltered to find available emotions
        unfiltered = self.loader.get_study_metadata(1)
        if unfiltered is not None and 'EMOTION' in unfiltered.columns:
            emotions = unfiltered['EMOTION'].unique().tolist()[:2]
            if len(emotions) > 0:
                filtered = self.loader.get_study_metadata(1, emotions=emotions)
                self.assertIsNotNone(filtered)


class TestPOPANEDataLoaderAdditional(unittest.TestCase):
    """Additional tests for POPANEDataLoader"""

    def setUp(self):
        self.loader = popane_data_loader.POPANEDataLoader(data_dir="data/raw/")

    def test_downloader_initialized(self):
        """Test that downloader is properly initialized"""
        self.assertIsNotNone(self.loader.downloader)
        from emotion.dataloader.downloader import POPANEDownloader
        self.assertIsInstance(self.loader.downloader, POPANEDownloader)

    def test_download_complete(self):
        """Test download_complete method delegates to downloader"""
        result = self.loader.download_complete()
        self.assertIsInstance(result, bool)

    def test_number_of_downloads(self):
        """Test number_of_downloads method delegates to downloader"""
        result = self.loader.number_of_downloads()
        self.assertIsInstance(result, int)

    def test_get_subject_ids(self):
        """Test getting subject IDs for a study"""
        subject_ids = self.loader.get_subject_ids(1)
        self.assertIsInstance(subject_ids, list)
        if len(subject_ids) > 0:
            self.assertIsInstance(subject_ids[0], (int, np.integer))

    def test_get_unique_emotions(self):
        """Test getting unique emotions for a study"""
        emotions = self.loader.get_unique_emotions(1)
        self.assertIsInstance(emotions, list)

    def test_data_dir_property(self):
        """Test data_dir property"""
        self.assertEqual(self.loader.data_dir, "data/raw/")

    def test_is_metadata_cached(self):
        """Test is_metadata_cached method"""
        result = self.loader.is_metadata_cached()
        self.assertIsInstance(result, bool)


# if __name__ == '__main__':
#     unittest.main()