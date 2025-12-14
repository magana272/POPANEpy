import dataclasses

import emotion.studies.dataloader.popaneloader as popane_data_loader
from emotion.studies.study import Subject

import unittest


class test_POPANEMETADATALoader(unittest.TestCase):
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
            study2_cols = list(map(lambda x: x.upper, ["subject_id", "sex", "age", "height", "weight", "stimuli1", "stimuli2",
                                               "study_name", "file_name", "emotion", "file_path"]))
            for col in study2_cols:
                self.assertIn(col, dataclasses.asdict(study2_meta).keys())
    def test_study3_metadata(self):
        study3_meta = self.loader.get_study_metadata(3)
        if study3_meta is not None:
            study3_cols = list(map(lambda x: x.upper, ["subject_id", "sex", "age", "height", "weight",
                                               "stimuli1", "stimuli2", "study_name", "file_name", "emotion", "file_path"]))
            for col in study3_cols:
                self.assertIn(col, study3_meta.columns)
            self.assertEqual(len(study3_meta.columns), 11)

    def test_study4_metadata(self):
        study4_meta = self.loader.get_study_metadata(4)
        if study4_meta is not None:
            study4_cols = list(map(str.upper, ["subject_id", "sex", "age", "height", "weight", "stimuli",
                                               "study_name", "file_name", "emotion", "file_path"]))
            for col in study4_cols:
                self.assertIn(col, study4_meta.columns)
            self.assertEqual(len(study4_meta.columns), 10)

    def test_study5_metadata(self):
        study5_meta = self.loader.get_study_metadata(5)
        if study5_meta is not None:
            study5_cols = list(map(str.upper, ["subject_id", "sex", "age", "height", "weight",
                                               "stimuli1", "stimuli2", "stimuli3", "study_name",
                                               "file_name", "emotion", "file_path"]))
            for col in study5_cols:
                self.assertIn(col, study5_meta.columns)
            self.assertEqual(len(study5_meta.columns), 12)

    def test_study6_metadata(self):
        study6_meta = self.loader.get_study_metadata(6)
        if study6_meta is not None:
            study6_cols = list(map(str.upper, ["subject_id", "sex", "age", "height", "weight", "stimuli1", "stimuli2",
                                               "stimuli3", "stimuli4", "stimuli5", "stimuli6",
                                               "study_name", "file_name", "emotion", "file_path"]))
            for col in study6_cols:
                self.assertIn(col, study6_meta.columns)
            self.assertEqual(len(study6_meta.columns), 15)

    def test_study7_metadata(self):
        study7_meta = self.loader.get_study_metadata(7)
        if study7_meta is not None:
            study7_cols = list(map(str.upper, ["subject_id", "sex", "age", "height", "weight", "stimuli1", "stimuli2",
                                               "stimuli3", "stimuli4", "stimuli5",
                                               "study_name", "file_name", "emotion", "file_path"]))
            for col in study7_cols:
                self.assertIn(col, study7_meta.columns)
            self.assertEqual(len(study7_meta.columns), 14)

    def test_get_data_for_subject_from_study(self):
        subject_data: Subject |None = self.loader.get_data_for_subject_from_study(1, 1)
        if subject_data is not None:
            subject_data_df = subject_data.to_dataframe()
            self.assertIn('TIMESTAMP', subject_data_df.columns)
            self.assertIn('AFFECT', subject_data_df.columns)
            self.assertIn('ECG', subject_data_df.columns)
            self.assertIn('EDA', subject_data_df.columns)
            self.assertIn('TEMP', subject_data_df.columns)
            self.assertIn('RESPIRATION', subject_data_df.columns)
            self.assertIn('SBP', subject_data_df.columns)
            self.assertIn('DBP', subject_data_df.columns)
            self.assertIn('MARKER', subject_data_df.columns)
        self.assertIsNotNone(subject_data)
# if __name__ == '__main__':
#     unittest.main()