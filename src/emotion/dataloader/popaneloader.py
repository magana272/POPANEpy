"""

Module to load and handle
for multiple emotion studies.

"""
from importlib.resources import files
from fileinput import filename
from os import listdir
from os.path import isfile
from os.path import join
import pickle
import re
import stat
from tkinter import E
from turtle import st
from typing import Any, DefaultDict, Generator, cast
from matplotlib import use
import pandas as pd
import polars as pl
import requests
import os
import threading
from time import sleep
import zipfile

from scipy.stats import t
from test.test_ensurepip import Traversable
# Define expected columns for each study
STUDY1_COLUMNS = set(["TIMESTAMP", "AFFECT", "ECG",
                      "EDA", "TEMP", "RESPIRATION", "SBP", "DBP", "MARKER"])
STUDY2_COLUMNS = set(["TIMESTAMP", "AFFECT", "ECG",
                      "EDA", "SBP", "DBP", "CO", "TPR", "MARKER"])
STUDY3_COLUMNS = set(["TIMESTAMP", "AFFECT",
                      "ECG", "EDA", "SBP", "DBP", "CO", "TPR", "MARKER"])
STUDY4_COLUMNS = set(["TIMESTAMP", "ECG", "EDA", "SBP",
                     "DBP", "CO", "TPR", "MARKER"])
STUDY5_COLUMNS = set(["TIMESTAMP", "AFFECT", "ECG", "EDA",
                     "SBP", "DBP", "CO", "TPR", "MARKER"])
STUDY6_COLUMNS = set(["TIMESTAMP", "AFFECT", "ECG", "DZDT", "DZ",
                      "Z0", "EDA", "SBP", "DBP", "CO", "TPR", "MARKER"])
STUDY7_COLUMNS = set(["TIMESTAMP", "AFFECT", "ECG", "DZDT",
                      "DZ", "Z0", "MARKER"])


class POPANEDataLoader:
    """Class to handle metadata
    for multiple studies.
    """
    metadata_path: str
    data_dir: str
    studies: dict[int, pd.DataFrame | None] = {1: None,
                                               2: None,
                                               3: None,
                                               4: None,
                                               5: None,
                                               6: None,
                                               7: None}
    column_names: dict[int, set[str]] = {1: STUDY1_COLUMNS,
                                         2: STUDY2_COLUMNS,
                                         3: STUDY3_COLUMNS,
                                         4: STUDY4_COLUMNS,
                                         5: STUDY5_COLUMNS,
                                         6: STUDY6_COLUMNS,
                                         7: STUDY7_COLUMNS}
    stimuli: pd.DataFrame | None = None

    __tempdir: Traversable = files('emotion') / 'tmp'

    lock = threading.Lock()
    __downloads_completed: bool = False
    __number_of_downloads: int = 0
    zip_lock = threading.Lock()
    completed_files: list[str] = []

    def __init__(self,
                 #  metadata_path: str| None = "./data/raw/metadata.xlsx",
                 data_dir: str | None = "./data/raw/"):

        self.metadata_path = cast(str, files(
            'emotion') / 'meta' / 'metadata_R1.xlsx')
        if data_dir is not None:
            self.data_dir = data_dir
        if (self.metadata_path is not None) and (data_dir is not None):
            self.load_data()

    def load_data(self):
        """Load metadata from the Excel file and store it in class attributes."""
        if os.path.exists(os.path.join(cast(str, self.__tempdir), "studies.pkl")):
            with open(os.path.join(cast(str, self.__tempdir), "studies.pkl"), 'rb') as f:
                self.studies = pickle.load(f)
            if os.path.exists(os.path.join(cast(str, self.__tempdir), "stimuli.pkl")):
                with open(os.path.join(cast(str, self.__tempdir), "stimuli.pkl"), 'rb') as f:
                    self.stimuli = pickle.load(f)
            return
        sheetnames = pd.ExcelFile(self.metadata_path).sheet_names
        sheet1cols = ["ID", "sex", "age", "height", "weight", "stimuli"]
        sheet2cols = ["ID", "sex", "age", "height",
                      "weight", "stimuli1", "stimuli2"]
        sheet4cols = sheet1cols
        sheet5cols = ["ID", "sex", "age", "height",
                      "weight", "stimuli1", "stimuli2", "stimuli3"]
        sheet6cols = ["ID", "sex", "age", "height", "weight", "stimuli1",
                      "stimuli2", "stimuli3", "stimuli4", "stimuli5", "stimuli6"]
        sheet7cols = ["ID", "age", "sex", "height", "weight",
                      "stimuli1", "stimuli2", "stimuli3", "stimuli4", "stimuli5"]
        studiesmeta = {}
        for sheet in sheetnames:
            if sheet == "study5":
                df = pd.read_excel(self.metadata_path,
                                   sheet_name=sheet, skiprows=7, usecols=sheet5cols)
            elif sheet in ["study6", "study7"]:
                df = pd.read_excel(self.metadata_path,
                                   sheet_name=sheet, skiprows=6, usecols=sheet6cols if sheet == "study6" else sheet7cols)
            elif sheet == 'list of stimuli':
                df = pd.read_excel(self.metadata_path,
                                   sheet_name=sheet, skiprows=1, nrows=33)
            elif sheet in ['study2', 'study3']:
                df = pd.read_excel(self.metadata_path,
                                   sheet_name=sheet, usecols=sheet2cols, skiprows=6)
            else:
                df = pd.read_excel(self.metadata_path,
                                   sheet_name=sheet, usecols=sheet1cols, skiprows=6)

            df.rename(columns={'ID': 'Subject_ID'}, inplace=True)
            df.columns = df.columns.str.upper()
            studiesmeta[sheet] = df
        self.studies[1] = studiesmeta.get('study1')
        self.studies[2] = studiesmeta.get('study2')
        self.studies[3] = studiesmeta.get('study3')
        self.studies[4] = studiesmeta.get('study4')
        self.studies[5] = studiesmeta.get('study5')
        self.studies[6] = studiesmeta.get('study6')
        self.studies[7] = studiesmeta.get('study7')
        self.stimuli = studiesmeta.get('list of stimuli')
        if self.stimuli is not None:
            self.stimuli.rename(
                columns={'STIMULI ID': 'STIMULI'}, inplace=True)
            self.stimuli.columns = self.stimuli.columns.str.upper()

        self.__addstudy_tostudy_df()
        for i, (study, df) in enumerate(self.studies.items()):
            if df is None or df.empty:
                raise ValueError(f"Study {study} metadata is empty or None.")

        self.__join_file_names()

        with open(os.path.join(cast(str, self.__tempdir), "studies.pkl"), 'wb') as f:
            for study_number in range(1, 8):
                df = self.studies.get(study_number)
                if df is None or df.empty:
                    raise ValueError(
                        f"Study {study_number} metadata is empty or None.")
            pickle.dump(self.studies, f)
        if self.stimuli is not None:
            with open(os.path.join(cast(str, self.__tempdir), "stimuli.pkl"), 'wb') as f:
                pickle.dump(self.stimuli, f)

    def get_unique_emotions(self, study_number) -> list[str]:
        """Get a list of unique emotions present in the specified study."""
        study_meta = self.get_study_metadata(study_number)
        if study_meta is None:
            return []
        unique_emotions = study_meta['EMOTION'].unique().tolist()
        return unique_emotions

    def get_subject_ids(self, study_number) -> list[int]:
        """Get a list of subject IDs for a specific study."""
        study_meta = self.get_study_metadata(study_number)
        if study_meta is None:
            return []
        # print(study_meta.head())
        subject_ids = study_meta['SUBJECT_ID'].unique().tolist()
        return subject_ids

    def get_study_metadata(self, study_number) -> pd.DataFrame | None:
        """Retrieve metadata for a specific study based on the study number."""
        return self.studies.get(study_number)

    def get_data_for_subject_from_study_lazy(
            self, study_number: int, subject_id: str, measurements: tuple[str, ...],
            dtypes: dict[str, str]) -> Generator[pd.DataFrame, None, None]:
        """Yield dataframes for a specific subject from a study."""

        study_meta = self.get_study_metadata(study_number)
        print(
            f"Getting data for subject {subject_id} from study {study_number} lazily.")
        if study_meta is None:
            print(f"Study {study_number} metadata not found.")
            raise Exception("Study metadata not found.")

        print(f"SubjectId: {subject_id}")
        subject_meta = study_meta.loc[
            study_meta["SUBJECT_ID"] == int(subject_id), ["FILE_PATH", "EMOTION", "FILE_NAME"]]
        print(f"Subject metadata found with {len(subject_meta)} files.")
        measurements_dtypes = {k: v for k,
                               v in dtypes.items() if k in measurements}

        for row in subject_meta.itertuples(index=False):
            print(row)
            data = pd.read_csv(
                cast(str, row.file_path),
                skiprows=9,
                low_memory=False,
                usecols=measurements,   # add if possible
                # add if possible
                dtype=cast(DefaultDict[str, str], measurements_dtypes)
            )
            yield data

    def get_data_for_subject_from_study(self, study_number, subject_id):
        """Get data for a specific subject from a specified study."""
        res = None
        study_meta = self.get_study_metadata(study_number)
        print(
            f"Getting data for subject {subject_id} from study {study_number}.")
        if study_meta is None:
            print(f"Study {study_number} metadata not found.")
            return None
        subject_meta = study_meta[study_meta['SUBJECT_ID'] == int(subject_id)]
        meta_columns = ["FILE_PATH", "EMOTION", "FILE_NAME"]
        subject_files = subject_meta[meta_columns].values.tolist()
        for file_path, emotion, file_name in subject_files:
            data = pd.read_csv(file_path, skiprows=9)
            print(data)
            data.columns = data.columns.str.upper()
            if res is None:
                res = data
                res["STUDY_NAME"] = f"study{study_number}"
                res["SUBJECT_ID"] = subject_id
                res["EMOTION"] = emotion
                res["FILE_NAME"] = file_name
                res["FILE_PATH"] = file_path
            else:
                data["STUDY_NAME"] = f"study{study_number}"
                data["SUBJECT_ID"] = subject_id
                data["EMOTION"] = emotion
                data["FILE_NAME"] = file_name
                data["FILE_PATH"] = file_path
                res = pd.concat([res, data], ignore_index=True)
        if res is not None:
            res.columns = res.columns.str.upper()
        return res

    def get_unique_emotions_for_all_studies(self):
        """Get a list of all unique 
        emotions across all studies.
        """
        all_unique_emotions = set()
        for study_number in range(1, 8):
            unique_emotions = self.get_unique_emotions(study_number)
            all_unique_emotions |= set(unique_emotions)
        return list(set(all_unique_emotions))

    def get_data_from_study_with_emotions(self, study_number, emotions):
        """Get data from a specified study for given emotions."""
        study_meta = self.get_study_metadata(study_number)
        if study_meta is None:
            return pl.DataFrame()
        rows = study_meta[study_meta["EMOTION"].isin(emotions)][
            ["FILE_PATH", "SUBJECT_ID", "FILE_NAME", "EMOTION"]
        ].values.tolist()
        dfs = []
        study_name = f"study{study_number}"
        for file_path, subject_id, file_name, emotion in rows:
            data = pl.read_csv(file_path, skip_rows=11,
                               infer_schema_length=10000)
            numeric_columns = [col for col in data.columns
                               if col not in ['MARKER', 'STUDY_NAME', 'SUBJECT_ID',
                                              'EMOTION', 'FILE_NAME', 'FILE_PATH']]
            cast_exprs = [pl.col(col).cast(pl.Float64, strict=False)
                          for col in numeric_columns]
            data = data.with_columns(cast_exprs)
            data = data.with_columns([
                pl.lit(study_name).alias("STUDY_NAME"),
                pl.lit(subject_id).alias("SUBJECT_ID"),
                pl.lit(emotion).alias("EMOTION"),
                pl.lit(file_name).alias("FILE_NAME"),
                pl.lit(file_path).alias("FILE_PATH")
            ])
            dfs.append(data)
        if len(dfs) == 0:
            return pl.DataFrame()
        return pl.concat(dfs)

    def get_data_from_study_with_emotions_limit(self, study_number, emotions, limit):
        """Get data from a specified study for given emotions,
        limited to a certain number of unique files.
        """
        res = None
        study_meta = self.get_study_metadata(study_number)
        if study_meta is None:
            return None
        subject_meta = study_meta[study_meta['Emotion'].isin(
            emotions)][["FILE_PATH", "SUBJECT_ID", "FILE_NAME", "EMOTION"]]
        subject_meta = subject_meta.values.tolist()
        for file_path, subject_id, file_name, emotion in subject_meta:
            data = pd.read_csv(file_path, skiprows=11)
            if res is None:
                res = data
                res["STUDY_NAME"] = f"study{study_number}"
                res["SUBJECT_ID"] = subject_id
                res["EMOTION"] = emotion
                res["FILE_NAME"] = file_name
                res["FILE_PATH"] = file_path
            else:
                data["STUDY_NAME"] = f"study{study_number}"
                data["SUBJECT_ID"] = subject_id
                data["EMOTION"] = emotion
                data["FILE_NAME"] = file_name
                data["FILE_PATH"] = file_path
                res = pd.concat([res, data], ignore_index=True)
            if res['FILE_NAME'].nunique() >= limit:
                break
        if res is not None:
            res.columns = res.columns.str.upper()
        return res

    def get_overlap_features_across_studies(self, study_numbers: list[int]) -> set[Any]:
        """Get overlapping features across specified studies."""
        set_list: list[set[str] | None] = [
            self.column_names.get(num) for num in study_numbers]
        valid_sets = [s for s in set_list if s is not None]
        if not valid_sets:
            return set()
        overlap_features = set.intersection(*valid_sets)
        return overlap_features

    def get_features_for_study(self, study_number: int) -> set[str] | None:
        """Get features for a specific study."""
        return self.column_names.get(study_number)

    def get_all_features(self):
        """Get all unique features across all studies."""
        all_features = set()
        for study_number in range(1, 8):
            features = self.get_features_for_study(study_number)
            if features is not None:
                all_features.update(features)
        return list(all_features)

    def download_complete(self) -> bool:
        """Check if all downloads are complete."""
        with self.lock:
            return self.__downloads_completed

    def number_of_downloads(self) -> int:
        """Get the number of completed downloads."""
        with self.lock:
            return self.__number_of_downloads

    def unzip_file(self, zip_path: str, extract_to: str) -> None:
        """Unzip a file to the specified directory."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

    def download_data(self, directory: str = "./data/raw/") -> None:
        """Download data from OSF storage."""
        data = ["https://data.psychosensing.psnc.pl/popane/files/study1.zip",
                "https://data.psychosensing.psnc.pl/popane/files/study2.zip",
                "https://data.psychosensing.psnc.pl/popane/files/study3.zip",
                "https://data.psychosensing.psnc.pl/popane/files/study4.zip",
                "https://data.psychosensing.psnc.pl/popane/files/study5.zip",
                "https://data.psychosensing.psnc.pl/popane/files/study6.zip",
                "https://data.psychosensing.psnc.pl/popane/files/study7.zip",
                "https://data.psychosensing.psnc.pl/popane/files/metadata.xlsx"]
        print(f"Current working directory: {os.getcwd()}")

        os.makedirs(directory, exist_ok=True)
        data_size_string = """
        ----------------------------------------------
        |File	    |Size of .zip	|Size unzipped   |
        |---------------------------------------------
        |study1.zip	|1.7 GB	        |9.1 GB          |
        |study2.zip	|1.6 GB	        |14.5 GB         |
        |study3.zip	|2.6 GB	        |24.6 GB         |
        |study4.zip	|0.6 GB	        |6.9 GB          |
        |study5.zip	|1.5 GB	        |18.5 GB         |
        |study6.zip	|2.7 GB	        |31.7 GB         |
        |study7.zip	|7.0 GB	        |50.4 GB         |
        ----------------------------------------------
        Total size (unzipped): 155.7 GB
        """
        print("The following files will be downloaded:")
        print(data_size_string)
        answer = input("Are you sure you want to download? (y/n): ")
        if answer.lower() != 'y':
            print("Download cancelled.")
            return
        import joblib
        joblib.parallel.DEFAULT_BACKEND = 'threading'
        threading_list = []
        with joblib.parallel_config('threading', n_jobs=20):
            for url in data:
                filename = url.split("/")[-1]
                filepath = os.path.join(directory, filename)
                print(f"Downloading {filename}...")
                threading_list.append(threading.Thread(target=self.threaded_download,
                                                       args=(filename, url, filepath), daemon=True))
            threading_list.append(threading.Thread(
                target=self.check_all_downloads, args=(directory,), daemon=True))
            for thread in threading_list:
                thread.start()
            for thread in threading_list:
                thread.join()
            zip_thread_list = []
            for file in self.completed_files:
                if file.endswith('.zip'):
                    extract_to = os.path.splitext(file)[0]
                    t = threading.Thread(target=self.unzip_file, args=(
                        file, extract_to), daemon=True)
                    zip_thread_list.append(t)
            for thread in zip_thread_list:
                thread.start()
            for thread in zip_thread_list:
                thread.join()

        return

    # Private methods

    def __addstudy_tostudy_df(self) -> None:
        for i in range(1, 8):
            df = self.studies.get(i)
            if df is None:
                continue
            df['STUDY_NAME'] = f'STUDY{i}'
            self.studies[i] = df

    def __get_all_file_paths(self) -> list[str]:
        files = []
        for d in listdir(self.data_dir):
            if os.path.isdir(f"{self.data_dir}{d}"):
                print(f"Scanning directory: {self.data_dir}{d}")
                files += [f"{self.data_dir}{d}/{f}" for
                          f in listdir(f"{self.data_dir}{d}")
                          if isfile(join(f"{self.data_dir}{d}", f))]
        return files

    def __get_file_names(self, files):
        print("Extracting file names and metadata...")
        subject_meta = dict(STUDY_NAME=[],
                            SUBJECT_ID=[],
                            SUBJECT_AGE=[],
                            SUBJECT_SEX=[],
                            FILE_NAME=[],
                            FILE_PATH=[])
        for filepath in files:
            with open(filepath, newline="\n", encoding="utf-8") as input_file:
                head = [input_file.readline()
                        .replace("#", "")
                        .strip("\n")
                        .split(",")
                        for _ in range(4)]
                head.append(["FILE_NAME",
                             re.findall(r"/([a-zA-Z_0-9]*).csv", string=filepath)[0].upper()])
                head.append(["FILE_PATH", filepath])
                for _, list_item in enumerate(head):
                    key = list_item[0].upper()
                    vals = [val for val in list_item[1:]]
                    if key in subject_meta:
                        subject_meta[key.upper()].extend(vals)
        metadata_df = pd.DataFrame.from_dict(subject_meta)
        return metadata_df

    def __clean_meatadata_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if (df is None) or (df.empty):
            return pd.DataFrame()
        df.loc[(df.STUDY_NAME == "STUDY 6") &
               (df.SUBJECT_SEX == "1"),
               "SUBJECT_SEX"] = 0
        df.loc[(df.STUDY_NAME == "STUDY 6") &
               (df.SUBJECT_SEX == "2"),
               "SUBJECT_SEX"] = 1
        df["EMOTION"] = df["FILE_NAME"].str.extract(
            r'_([a-zA-Z_]*)[0-9]*?$')[0]
        df["STUDY_NAME"] = df["STUDY_NAME"].str.upper().str.replace(" ", "")
        df.SUBJECT_ID = df.SUBJECT_ID.astype(int)
        meta_columns = ["STUDY_NAME", "SUBJECT_ID",
                        "FILE_PATH", "FILE_NAME", "EMOTION"]
        df = df[df["EMOTION"] != "All"]
        df = df[meta_columns]
        df.columns = df.columns.str.upper()
        return df

    def __join_file_names(self) -> None:
        files = self.__get_all_file_paths()

        metadata_df = self.__get_file_names(files)
        metadata_df = self.__clean_meatadata_df(metadata_df)
        meta_columns = metadata_df.columns.tolist()
        for study_number in range(1, 8):
            study = self.studies.get(study_number)
            if study is not None:
                study.columns = study.columns.str.upper()
                self.studies[study_number] = study
        for _, (study_number, study) in enumerate(self.studies.items()):
            if study is None:
                continue
            if not metadata_df[meta_columns].empty:
                self.studies[study_number] = study.merge(metadata_df[meta_columns], on=[
                                                         "SUBJECT_ID", "STUDY_NAME"], how="left")
            else:
                raise ValueError(
                    "Metadata dataframe is empty, cannot merge file names.")

    def __update_number_of_downloads(self, filename: str, filepath: str) -> None:
        with self.lock:
            self.__number_of_downloads += 1
            if self.__number_of_downloads == 8:
                self.__downloads_completed = True
            if filename.endswith('.zip'):
                self.completed_files.append(filepath)
            print(f"Total downloads completed: {self.__number_of_downloads}")

    def threaded_download(self, filename, url, filepath) -> None:
        if os.path.exists(filepath):
            print(f"{filename} already exists. Skipping download.")
            self.__update_number_of_downloads(filename, filepath)
            return
        response = requests.get(url, stream=True)
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        self.__update_number_of_downloads(filename, filepath)

    def check_all_downloads(self, dir):
        print("Checking download progress...")
        threads = []
        while not self.download_complete():
            print("Download progress:")
            COMPLETESIZE = 1.7 + 1.6 + 2.6 + 0.6 + 1.5 + 2.7 + 7.0
            print(f"Total size to download: {COMPLETESIZE} GB")
            total_downloaded = 0.0
            for filename in os.walk(dir):
                for file in filename[2]:
                    if file.endswith('.zip'):
                        filepath = os.path.join(filename[0], file)
                        filesize = os.path.getsize(
                            filepath) / (1024 * 1024 * 1024)  # Convert to GB
                        total_downloaded += filesize
            print(f"Total downloaded so far: {total_downloaded:.2f} GB")
            print(f"{total_downloaded/COMPLETESIZE*100:.2f}% completed")
        for thread in threads:
            thread.join()
