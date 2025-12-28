"""

Module to load and handle metadata
for multiple emotion studies.

"""
from __future__ import annotations

import os
import re
import sys
from importlib.resources import files
from io import BufferedWriter
from os import listdir
from os.path import isfile
from os.path import join
from pickle import load, dump
from typing import cast, TYPE_CHECKING

# Handle Traversable import for different Python versions
try:
    from importlib.resources.abc import Traversable
except (ImportError, ModuleNotFoundError):
    try:
        from importlib.abc import Traversable
    except ImportError:
        # Fallback for typing purposes
        if TYPE_CHECKING:
            from typing import Any as Traversable
        else:
            Traversable = object

import pandas as pd
import polars as pl
from polars import DataFrame

from emotion.dataloader.downloader import POPANEDownloader


class POPANEMETADataLoader:
    """Class to handle metadata
    for multiple studies.
    """
    metadata_path: str
    __tempdir: Traversable = files('emotion') / 'tmp'
    studies: dict[int, DataFrame | None] = {
        1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None}
    stimuli: pd.DataFrame | None = None

    def __init__(self, studies_stimuli: tuple[dict[int, DataFrame | None], pd.DataFrame | None] | None = None) -> None:
        if studies_stimuli is not None:
            self.studies, self.stimuli = studies_stimuli[0], studies_stimuli[1]

        elif self.is_cached():
            with open(os.path.join(cast(str, self.__tempdir), "studies.pkl"), 'rb') as f:
                self.studies = load(f)
            with open(os.path.join(cast(str, self.__tempdir), "stimuli.pkl"), 'rb') as f:
                self.stimuli = load(f)
        else:

            self.metadata_path = cast(str, files(
                'emotion') / 'meta' / 'metadata_R1.xlsx')
            self.load_metadata()

    def is_cached(self) -> bool:
        """Check if metadata is cached."""
        return os.path.exists(os.path.join(cast(str, self.__tempdir), "studies.pkl"))

    # noinspection PyTypeChecker
    def load_metadata(self):

        sheet_info = {"study1": ["ID", "sex", "age", "height", "weight", "stimuli"],
                      "study2": ["ID", "sex", "age", "height", "weight", "stimuli1", "stimuli2"],
                      "study3": ["ID", "sex", "age", "height", "weight", "stimuli1", "stimuli2"],
                      "study4": ["ID", "sex", "age", "height", "weight", "stimuli"],
                      "study5": ["ID", "sex", "age", "height", "weight", "stimuli1", "stimuli2", "stimuli3"],
                      "study6": ["ID", "sex", "age", "height", "weight", "stimuli1", "stimuli2", "stimuli3", "stimuli4",
                                 "stimuli5", "stimuli6"],
                      "study7": ["ID", "age", "sex", "height", "weight", "stimuli1", "stimuli2", "stimuli3", "stimuli4",
                                 "stimuli5"],
                      "list of stimuli": None}
        studies_meta = {}
        for sheet, cols in sheet_info.items():
            if sheet == "study5":
                df = pd.read_excel(self.metadata_path,
                                   sheet_name=sheet, skiprows=7, usecols=cols)
            elif sheet in ["study6", "study7"]:
                df = pd.read_excel(self.metadata_path, sheet_name=cast(
                    str, sheet), skiprows=6, usecols=cols)
            elif sheet == "list of stimuli":
                df = pd.read_excel(self.metadata_path,
                                   sheet_name=sheet, skiprows=1, nrows=33)
            elif sheet in ['study2', 'study3']:
                df = pd.read_excel(self.metadata_path,
                                   sheet_name=sheet, usecols=cols, skiprows=6)
            else:
                df = pd.read_excel(self.metadata_path,
                                   sheet_name=sheet, usecols=cols, skiprows=6)
            studies_meta[sheet] = df
        self.__add_study_name(studies_meta)
        self.__clean_metadata_df(studies_meta)
        for study_number in range(1, 8):
            df = studies_meta.get(f'study{study_number}')
            if df is None:
                raise Exception("DataFrame is None")
            self.set_study_metadata(study_number, df)
        for key, val in self.studies.items():
            if val is None:
                raise Exception("DataFrame is None")

    def __add_study_name(self, studies_meta):
        for i in range(1, 8):
            study = studies_meta.get(f'study{i}')
            study["STUDY_NAME"] = f"STUDY{i}"
            studies_meta[i] = study
        return studies_meta

    def get_stimuli(self) -> pd.DataFrame | None:
        """Get stimuli metadata."""
        return self.stimuli

    def set_stimuli(self, stimuli: pd.DataFrame) -> None:
        """Set stimuli metadata."""
        self.stimuli = stimuli

    def get_study_metadata(self, study_number: int, emotions: list[str] | None = None) -> pd.DataFrame | None:
        """Get metadata for a specific study."""
        metadata_df = self.studies.get(study_number)
        if metadata_df is None:
            return None
        if emotions is not None:
            metadata_df = metadata_df[metadata_df['EMOTION'].isin(emotions)]  # type: ignore
        return metadata_df # pyright: ignore[reportReturnType]

    def set_study_metadata(self, study_number: int, df: DataFrame) -> pd.DataFrame | None:
        """Set metadata for a specific study."""
        self.studies[study_number] = df

    def get_all_studies_metadata(self) -> dict[int, pd.DataFrame | None]:
        """Get metadata for all studies."""
        return {study_number: self.get_study_metadata(study_number) for study_number in range(1, 8)}

    def get_stimuli_metadata(self) -> pd.DataFrame | None:
        """Get stimuli metadata."""
        return self.stimuli

    def __clean_metadata_df(self, studies_meta: dict[str, pd.DataFrame]) -> None:
        stimuli = studies_meta.get('list of stimuli')
        if stimuli is not None:
            print(stimuli.columns[0])
            stimuli = stimuli.rename(columns={stimuli.columns[0]: 'STIMULI'}, errors='raise')
            stimuli.columns = stimuli.columns.str.upper()

            self.set_stimuli(stimuli)
        else:
            raise Exception("No stimuli found")
        for key, val in studies_meta.items():
            val.columns = val.columns.str.upper()
            val.rename(columns={"ID": "SUBJECT_ID"}, inplace=True)
            self.set_study_metadata(key, val)  # type: ignore

    def get_subject_metadata(self, study_number, subject_id) -> pd.DataFrame | None:
        study_frame = self.studies.get(study_number)
        if study_frame is None:
            return None
        study_frame = study_frame[study_frame['SUBJECT_ID'] == subject_id]
        return study_frame


class POPANEDataLoader:
    """Class to handle metadata and data loading for multiple studies.

    Uses POPANEDownloader for downloading functionality.
    """

    data_dir: str
    studies_meta_loader: POPANEMETADataLoader
    downloader: POPANEDownloader

    __tempdir: Traversable = files('emotion') / 'tmp'

    def __init__(self, data_dir: str | None = "./data/raw/"):
        """Initialize the data loader.

        Args:
            data_dir: Directory containing raw study data.
        """
        if data_dir is not None:
            self.data_dir = data_dir

        # Initialize the downloader for download operations
        self.downloader = POPANEDownloader(data_dir=self.data_dir)

        if self.is_metadata_cached():
            self.studies_meta_loader = self.read_metadata_from_cache()
        else:
            self.studies_meta_loader = POPANEMETADataLoader()
            self.load_data()

    def load_data(self):

        def __get_all_file_paths() -> list[str]:
            filez = []
            for d in listdir(self.data_dir):
                if os.path.isdir(f"{self.data_dir}{d}"):
                    filez += [f"{self.data_dir}{d}/{f}" for
                              f in listdir(f"{self.data_dir}{d}")
                              if isfile(join(f"{self.data_dir}{d}", f))]
            return filez

        def __join_metadata_file_with_header_information(df) -> None:
            meta_columns = df.columns.tolist()
            for study_number in range(1, 8):
                study = self.studies_meta_loader.get_study_metadata(study_number)
                if study is None:
                    raise Exception(
                        f"Metadata for study {study_number} is None.")
                if not df[meta_columns].empty:
                    self.studies_meta_loader.set_study_metadata(study_number,
                                                                study.merge(df[meta_columns],
                                                                            on=[
                                                                                "SUBJECT_ID", "STUDY_NAME"],
                                                                            how="left"))  # type: ignore
                else:
                    raise Exception(
                        "Metadata dataframe is empty, cannot merge file names.")

        def __clean_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
            if df is None:
                raise Exception("Metadata dataframe is None or empty.")
            df.loc[(df.STUDY_NAME == "STUDY 6") & (df.SUBJECT_SEX == "1"),
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
            df = df[df["EMOTION"] != "ALL"]
            df = df[meta_columns]
            df.columns = df.columns.str.upper()
            return df

        def __parse_header_of_data_files(file_list):
            print("Extracting file names and metadata...")
            subject_meta = dict(STUDY_NAME=[],
                                SUBJECT_ID=[],
                                SUBJECT_AGE=[],
                                SUBJECT_SEX=[],
                                FILE_NAME=[],
                                FILE_PATH=[])
            for filepath in file_list:
                abs_filepath = os.path.abspath(filepath)
                with open(abs_filepath, newline="\n", encoding="utf-8") as input_file:
                    head = [input_file.readline()
                            .replace("#", "")
                            .strip("\n")
                            .split(",")
                            for _ in range(4)]
                    head.append(["FILE_NAME",
                                 re.findall(r"/([a-zA-Z_0-9]*).csv", string=abs_filepath)[0].upper()])
                    head.append(["FILE_PATH", abs_filepath])
                    for _, list_item in enumerate(head):
                        key = list_item[0].upper()
                        vals = [val for val in list_item[1:]]
                        if key in subject_meta:
                            subject_meta[key.upper()].extend(vals)
            metadata_df = pd.DataFrame.from_dict(subject_meta)
            return metadata_df

        files_from_path = __get_all_file_paths()
        header_info_df = __parse_header_of_data_files(files_from_path)
        header_info_df = __clean_metadata_df(header_info_df)
        __join_metadata_file_with_header_information(header_info_df)
        self.write_metadata_to_cache()

    def write_metadata_to_cache(self) -> None:
        """Write metadata to cache."""
        studies = self.studies_meta_loader.get_all_studies_metadata()
        stimuli = self.studies_meta_loader.get_stimuli_metadata()
        f: BufferedWriter
        with open(os.path.join(cast(str, self.__tempdir), "studies.pkl"), 'wb') as f:
            dump(studies, f)
        with open(os.path.join(cast(str, self.__tempdir), "stimuli.pkl"), 'wb') as f:
            dump(stimuli, f)

    def is_metadata_cached(self) -> bool:
        """Check if metadata is cached."""
        return os.path.exists(os.path.join(cast(str, self.__tempdir), "studies.pkl"))

    def read_metadata_from_cache(self) -> POPANEMETADataLoader:
        """Read metadata from cache."""
        with open(os.path.join(cast(str, self.__tempdir), "studies.pkl"), 'rb') as f:
            studies = load(f)
        with open(os.path.join(cast(str, self.__tempdir), "stimuli.pkl"), 'rb') as f:
            stimuli = load(f)
        return POPANEMETADataLoader((studies, stimuli))

    def download_complete(self) -> bool:
        """Check if all downloads are complete."""
        return self.downloader.download_complete()

    def number_of_downloads(self) -> int:
        """Get the number of completed downloads."""
        return self.downloader.number_of_downloads()

    def unzip_file(self, zip_path: str, extract_to: str) -> None:
        """Unzip a file to the specified directory."""
        self.downloader.unzip_file(zip_path, extract_to)

    def download_data(self, directory: str = "./data/raw/", studies=(1, 2, 3, 4, 5, 6, 7, "meta")) -> None:
        """Download data from remote storage.

        Args:
            directory: Directory to download to.
            studies: Tuple or list of study numbers to download (1-7 and/or "meta").
        """
        self.downloader.download_data(directory=directory, studies=studies)

    def unzip_files(self):
        """Unzip all completed zip files."""
        self.downloader.unzip_files()

    def get_study_metadata(self, study_number: int) -> pd.DataFrame | None:
        """Retrieve metadata for a specific study based on the study number."""
        return self.studies_meta_loader.get_study_metadata(study_number)

    def get_data_for_subject_from_study(self, study_number: int, subject_id: int) -> pd.DataFrame | None:
        """Get data for a specific subject from a specific study."""
        study_meta = self.studies_meta_loader.get_subject_metadata(study_number, subject_id)
        if study_meta is None:
            return None
        df = self.get_subject_data(study_number, subject_id)
        return df.to_pandas() if df is not None else None

    def get_subject_ids(self, study_number, emotion: list[str] | None = None) -> list[int]:
        """Get a list of subject IDs for a specific study."""
        study_meta = self.get_study_metadata(study_number)
        if study_meta is None:
            return []
        if study_meta.STUDY_NAME is not None:
            subject_ids = list(set([study_meta.SUBJECT_ID])) if isinstance(
                study_meta.SUBJECT_ID, int) else list(set(study_meta.SUBJECT_ID))
            return subject_ids
        return []

    def get_unique_emotions(self, study_number) -> list[str]:
        """Get a list of unique emotions present in the specified study."""
        study_meta = self.get_study_metadata(study_number)
        if study_meta is None:
            return []
        unique_emotions = list(set(study_meta.EMOTION))
        return unique_emotions

    def get_all_subjects_from_study(self, study_number: int, emotions: list[str] | None = None) -> dict[
        int, pd.DataFrame | None]:
        studymeta = self.studies_meta_loader.get_study_metadata(study_number, emotions)
        res: dict[int, pd.DataFrame | None] = {}
        if studymeta is None:
            return res
        if studymeta.SUBJECT_ID is not None:
            subject_ids = list(set([studymeta.SUBJECT_ID])) if isinstance(
                studymeta.SUBJECT_ID, int) else list(set(studymeta.SUBJECT_ID))
        else:
            subject_ids = list(set())
        for subject_id in subject_ids:
            res[subject_id] = self.get_data_for_subject_from_study(
                study_number, subject_id)
        return res

    def get_subject_metadata(self, study_number: int, subject_id: int) -> pd.DataFrame | None:
        """Get metadata for a specific subject in a specific study."""
        study_meta = self.studies_meta_loader.get_subject_metadata(
            study_number, subject_id)
        return study_meta

    def get_subject_data(self, study_id, subject_id) -> pl.DataFrame | None:
        """Get metadata DataFrame for a specific subject in a study."""
        metadata = self.get_subject_metadata(study_id, subject_id)
        dfs = []
        if metadata is None or metadata.FILE_PATH is None:
            return None
        for study_name, file_path, emotion, file_name, subject_id in list(
                zip(metadata.STUDY_NAME, metadata.FILE_PATH,  # pyright: ignore[reportGeneralTypeIssues]
                    metadata.EMOTION, metadata.FILE_NAME,  # type: ignore
                    metadata.SUBJECT_ID)):  # type: ignore
            data = pl.read_csv(file_path, skip_rows=9,
                               infer_schema_length=10000)

            numeric_columns = [col for col in data.columns
                               if col not in ['marker', 'Study_name', 'Subject_ID',
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

    def get_stimui(self) -> pd.DataFrame:
        return self.studies_meta_loader.get_stimuli() # type: ignore


def main():
    p = POPANEDataLoader()
    resp = p.get_subject_metadata(1, 1)
    print(resp)
    subject = p.get_data_for_subject_from_study(1, 1)


if __name__ == "__main__":
    main()
