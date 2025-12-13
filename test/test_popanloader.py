import emotion.dataloader.popaneloader as popane_data_loader


import unittest


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
        if study1_meta is not None:
            # ID	sex	age	height	weight	stimuli
            self.assertIn('SUBJECT_ID', study1_meta.columns)
            self.assertIn('AGE', study1_meta.columns)
            self.assertIn('SEX', study1_meta.columns)
            self.assertIn('HEIGHT', study1_meta.columns)
            self.assertIn('WEIGHT', study1_meta.columns)
            self.assertIn('STIMULI', study1_meta.columns)

    def test_study2_metadata(self):
        study2_meta = self.loader.get_study_metadata(2)
        if study2_meta is not None:
            study2_cols = list(map(str.upper, ["subject_id", "sex", "age", "height", "weight", "stimuli1", "stimuli2",
                                               "study_name", "file_name", "emotion", "file_path"]))
            for col in study2_cols:
                self.assertIn(col, study2_meta.columns)
            self.assertEqual(len(study2_meta.columns), 11)

    def test_study3_metadata(self):
        study3_meta = self.loader.get_study_metadata(3)
        if study3_meta is not None:
            study3_cols = list(map(str.upper, ["subject_id", "sex", "age", "height", "weight",
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
        subject_data = self.loader.get_data_for_subject_from_study(1, 1)
        self.assertIsNotNone(subject_data)
        if subject_data is not None:
            self.assertIn('TIMESTAMP', subject_data.columns)
            self.assertIn('ECG', subject_data.columns)
            self.assertIn('EDA', subject_data.columns)
            self.assertIn('TEMP', subject_data.columns)
            self.assertIn('RESPIRATION', subject_data.columns)
            self.assertIn('SBP', subject_data.columns)
            self.assertIn('DBP', subject_data.columns)
            self.assertIn('MARKER', subject_data.columns)
