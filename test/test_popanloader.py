"""
Comprehensive tests for POPANE emotion analysis package
"""
import os.path
import unittest
import dataclasses
import pandas as pd
import numpy as np

import emotion.dataloader.popaneloader as popane_data_loader
from emotion.studies.subject import Subject
from emotion.preprocessing.dataprocess import ECGSmoothTransformer, window_data
from emotion.visualization.figure import POPANEFigureGenerator
from emotion.models.random_forest import EmotionRandomForest
from emotion import POPANE


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
# if __name__ == '__main__':
#     unittest.main()