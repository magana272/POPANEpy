from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame


@dataclass
class POPANEMetadata:
    """Represents metadata for a study subject"""
    STUDY_NAME: list[str] | str
    SUBJECT_ID: list[int] | int
    AGE: list[int] | int
    SEX: list[int] | int
    HEIGHT: list[float] | float
    WEIGHT: list[float] | float
    EMOTION: list[str] | str
    STIMULI: list[str] | str | None
    STIMULI1: list[str] | str | None
    STIMULI2: list[str] | str | None
    STIMULI3: list[str] | str | None
    STIMULI4: list[str] | str | None
    STIMULI5: list[str] | str | None
    STIMULI6: list[str] | str | None
    STIMULI7: list[str] | str | None
    FILE_PATH: list[str] | str | None
    FILE_NAME: list[str] | str | None
    columns: list[str] | str | None


def create_popane_metadata(study_frame: DataFrame) -> pd.DataFrame | None:
    total_cols = ["STUDY_NAME", "SUBJECT_ID", "AGE", "SEX", "HEIGHT", "WEIGHT", "EMOTION", "STIMULI", "STIMULI1",
                  "STIMULI2",
                  "STIMULI3", "STIMULI4", "STIMULI5", "STIMULI6", "STIMULI7", "FILE_PATH", "FILE_NAME"]
    columns = [col for col in study_frame.columns if col in total_cols]
    return POPANEMetadata(STUDY_NAME=study_frame["STUDY_NAME"].to_list(),
                          SUBJECT_ID=study_frame["SUBJECT_ID"].to_list(),
                          AGE=study_frame['AGE'].to_list(),
                          SEX=study_frame['SEX'].to_list(),
                          HEIGHT=study_frame['HEIGHT'].to_list(),
                          WEIGHT=study_frame['WEIGHT'].to_list(),
                          EMOTION=study_frame['EMOTION'].to_list(),
                          STIMULI=study_frame['STIMULI'].to_list(
                          ) if "STIMULI" in study_frame.columns else None,
                          STIMULI1=study_frame['STIMULI1'].to_list(
                          ) if 'STIMULI1' in study_frame.columns else None,
                          STIMULI2=study_frame['STIMULI2'].to_list(
                          ) if 'STIMULI2' in study_frame.columns else None,
                          STIMULI3=study_frame['STIMULI3'].to_list(
                          ) if 'STIMULI3' in study_frame.columns else None,
                          STIMULI4=study_frame['STIMULI4'].to_list(
                          ) if 'STIMULI4' in study_frame.columns else None,
                          STIMULI5=study_frame['STIMULI5'].to_list(
                          ) if 'STIMULI5' in study_frame.columns else None,
                          STIMULI6=study_frame['STIMULI6'].to_list(
                          ) if 'STIMULI6' in study_frame.columns else None,
                          STIMULI7=study_frame['STIMULI7'].to_list(
                          ) if 'STIMULI7' in study_frame.columns else None,
                          FILE_PATH=study_frame[
                              'FILE_PATH'].to_list() if 'FILE_PATH' in study_frame.columns else None,
                          FILE_NAME=study_frame[
                              'FILE_NAME'].to_list() if 'FILE_NAME' in study_frame.columns else None
                          , columns=columns)
