import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from emotion.dataloader import POPANEDataLoader
from emotion.preprocessing.frequency import get_dominant_frequency, calculate_signal_energy
from emotion.visualization import POPANEFigureGenerator

FEATURES = ["ECG", "EDA", "SBP", "DBP", "respiration", "temp"]


class POPANEDataProcessor:
    """
    A class for processing POPANE emotion study data.
    """

    def __init__(self):
        pass


def window_data(dataset: pd.DataFrame,
                window__size=1000, steps=1000) -> tuple[np.ndarray, np.ndarray]:
    """Convert the dataset into overlapping windows for model training.
    Parameters:
    - dataset: The input dataset as a pandas DataFrame.
    - window_size: The size of each window (number of samples).
    - steps: The step size between consecutive windows.
    Returns:
    - X:
        A numpy array of shape (num_windows, window__size, num_features)
        containing the windowed data.
    - y:
        A numpy array of shape (num_windows,)
        containing the labels for each window.
    """
    df = dataset.copy()
    x = []
    y = []
    for _, group in df.groupby(["Subject_ID", "File_Name"]):
        data = group[FEATURES].to_numpy()
        label = group["Emotion"].iloc[0]
        n = data.shape[0]
        for start in range(0, n - window__size + 1, steps):
            end = start + window__size
            window = data[start:end]
            x.append(window)
            y.append(label)
    x = np.array(x)
    y = np.array(y)
    return x, y


class ECGSmoothTransformer(BaseEstimator, TransformerMixin):
    """
    Docstring for ECG_SmoothTransformer

    :var Parameters: Description
    :var Parameters: Description
    :var Returns: Description
    """

    def __init__(self, sigma):
        self.sigma = sigma
        self.n_features_in_ = None

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[-1]
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Apply Gaussian smoothing to the ECG signal in the dataset
        """
        X_out = X.copy()
        X_out["ECG"] = gaussian_filter1d(
            X["ECG"], sigma=self.sigma, axis=0, mode='nearest').astype(np.float32)
        return X_out


def visualize_transformations(study: int, subject_id: int, data_path: str, figsize: tuple = (15, 8),
                              start_end: tuple = (0, 10), sigma: int = 10):
    """
    Visualize the effect of different data transformations
    on physiological signalsfor a given subject.
        Parameters:
        - study: Study number (1-7)
        - subject_id: ID of the subject to visualize
        - figsize: Size of the figure
        - time_start: Start time for visualization
        - time_end: End time for visualization
        - sigma: Sigma value for Gaussian smoothing
    """

    popane_data_loader = POPANEDataLoader(data_path)
    subject = popane_data_loader.get_data_for_subject_from_study(
        study, subject_id)
    # subject = subject.t if subject is not None else None
    if subject is None:
        print(f"No data found for Study {study}, Subject_ID: {subject_id}")
        return
    time_start, time_end = start_end
    scaler = StandardScaler()
    smooth = ECGSmoothTransformer(sigma=sigma)
    features = ["ECG", "EDA", "SBP", "DBP"]
    if subject is None:
        print(f"No data found for Study {study}, Subject_ID: {subject_id}")
        return
    transformations = {
        "Original": subject[features],
        "Scaled": pd.DataFrame(scaler.fit_transform(subject[features]), columns=features),
        "Smoothed": pd.DataFrame(smooth.fit_transform(subject[features]), columns=features),
        "Scaled & Smoothed": None
    }
    transformations["Scaled & Smoothed"] = pd.DataFrame(
        smooth.fit_transform(transformations["Scaled"]), columns=features)
    for key, val in transformations.items():
        val["EMOTION"] = subject["EMOTION"]
        val["timestamp"] = subject["timestamp"]
        transformations[key] = val

    palette = sns.color_palette(
        "husl", n_colors=len(subject["EMOTION"].unique()))
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    for i, emotion in enumerate(subject["EMOTION"].unique()):
        emotion_mask = subject["EMOTION"] == emotion
        base_emotion = subject[emotion_mask].copy()
        for j, (transform_name, transform_data) in enumerate(transformations.items()):
            emotion_data = transform_data[emotion_mask].copy()
            emotion_data['time_offset'] = base_emotion['timestamp'] - \
                                          base_emotion['timestamp'].iloc[0]
            try:
                emotion_data = emotion_data.loc[(emotion_data["time_offset"] < time_end) & (
                        emotion_data["time_offset"] > time_start)].copy()
                POPANEFigureGenerator.plot_signals(emotion_data["time_offset"], emotion_data["ECG"],
                                                   title=transform_name, color=palette[i], axis=axes[j],
                                                   label=f"{emotion}")
            except Exception as e:
                print(emotion_data.columns)
                print(emotion_data.head())
                print(f"Error processing emotion data: {e}")
    axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9)
    fig.suptitle(
        f"Subject {subject_id} - Transformation Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_sigma_comparison(study, subject_id, sigmas=[1, 5, 10, 20, 50], figsize=(15, 8), time_start=None,
                               time_end=None):
    popane_data_loader = POPANEDataLoader()
    subject = popane_data_loader.get_data_for_subject_from_study(
        study, subject_id)

    if subject is None:
        print(f"No data found for Study {study}, Subject_ID: {subject_id}")
        return
    subject = subject.to_dataframe()
    palette = sns.color_palette("husl", n_colors=len(subject.Emotion.unique()))
    num_sigmas = len(sigmas)
    fig, axes = plt.subplots(num_sigmas, 1, figsize=figsize, sharex=True)
    if num_sigmas == 1:
        axes = [axes]
    for i, emotion in enumerate(subject["EMOTION"].unique()):
        subject_emotion = subject[subject["EMOTION"] == emotion].copy()
        subject_emotion['time_offset'] = subject_emotion['TIMESTAMP'] - \
                                         subject_emotion['TIMESTAMP'].iloc[0]
        subject_emotion = subject_emotion.loc[(subject_emotion["time_offset"] < time_end) & (
                subject_emotion["time_offset"] > time_start)].copy()

        for j, sigma in enumerate(sigmas):
            smooth = ECGSmoothTransformer(sigma=sigma)
            X_smoothed = pd.DataFrame(smooth.fit_transform(
                subject_emotion[["ECG", "EDA", "SBP", "DBP"]]), columns=['ECG', 'EDA', 'SBP', 'DBP'])
            t = subject_emotion["time_offset"].astype(np.float32).to_numpy()
            ecg = subject_emotion["ECG"].astype(np.float32).to_numpy()
            POPANEFigureGenerator.plot_signals(t, ecg,
                                               title=f"Sigma = {sigma}", color=palette[i], axis=axes[j],
                                               label=f"{emotion} (σ={sigma})")

    axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9)
    fig.suptitle(
        f"Subject {subject_id} - ECG Smoothing with Different Sigma Values", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def rolling_mean(series: pd.Series, window_size: int) -> pd.Series:
    return series.rolling(window=window_size, center=True).mean()


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def apply_rollingmean(df: pd.DataFrame, feature: str, window_size: int = 100) -> pd.DataFrame:
    df = df.copy()
    df[feature] = rolling_mean(df[feature], window_size=window_size)
    return df


def apply_rollingmedian(df: pd.DataFrame, feature: str, window_size: int = 100) -> pd.DataFrame:
    df = df.copy()
    df[feature] = df[feature].rolling(window=window_size, center=True).median()
    return df


def apply_ema(df: pd.DataFrame, feature: str, span: int = 20) -> pd.DataFrame:
    df[feature] = exponential_moving_average(df[feature], span=span)
    return df


def clean_DBP(subject_data: pd.DataFrame):
    emotions = subject_data["EMOTION"].unique()
    colors = sns.color_palette("husl", subject_data["EMOTION"].nunique())
    color_dict = {emotion: colors[i] for i, emotion in enumerate(emotions)}
    subject_id = subject_data["SUBJECT_ID"].unique()[0]
    fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    for subject_id in subject_data["SUBJECT_ID"].unique():
        subject_data = subject_data[subject_data["SUBJECT_ID"] == subject_id]
        for i, emotion in enumerate(subject_data["EMOTION"].unique()):
            emotion_data = subject_data[subject_data["EMOTION"] == emotion]
            emotion_data['time_offset'] = emotion_data['TIMESTAMP'] - \
                                          emotion_data['TIMESTAMP'].iloc[0]
            print(i, emotion, subject_data["FILE_NAME"].unique())
            POPANEFigureGenerator.plot_signals(emotion_data.time_offset, emotion_data.DBP,
                                               color=color_dict[emotion], label=emotion, axis=axs[0])
            median_cleaned = emotion_data.groupby("FILE_NAME", group_keys=False).apply(
                lambda x: apply_rollingmedian(x, 'DBP')).fillna(method="bfill").fillna(method="ffill")
            POPANEFigureGenerator.plot_signals(median_cleaned.time_offset, median_cleaned.DBP,
                                               color=color_dict[emotion], label=emotion, axis=axs[1])
    fig.legend(handles=[Line2D([0], [0], color=color_dict[emotion], lw=2, label=emotion) for emotion in
                        subject_data["EMOTION"].unique()],
               bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    axs[0].set_title(f"Subject {subject_id} - Raw DBP Signals")
    axs[1].set_title(f"Subject {subject_id} - Median Filtered DBP Signals")
    axs[1].set_xlabel("Time Offset (s)")
    plt.tight_layout()
    plt.show()
    return median_cleaned


def plot_transformations(df, time, signal, title, color, axis, label=None):
    emotions = df["EMOTION"].unique()
    colors = sns.color_palette("husl", df["EMOTION"].nunique())
    color_dict = {emotion: colors[i] for i, emotion in enumerate(emotions)}
    for subject_id in df["SUBJECT_ID"].unique():
        subject_data = df[df["SUBJECT_ID"] == subject_id]
        subject_data['time_offset'] = subject_data['TIMESTAMP'] - \
                                      subject_data['TIMESTAMP'].iloc[0]
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
        for i, emotion in enumerate(subject_data["EMOTION"].unique()):
            emotion_data = subject_data[subject_data["EMOTION"] == emotion]
            emotion_data = emotion_data[emotion_data["time_offset"] <= 10]
            POPANEFigureGenerator.plot_signals(emotion_data.time_offset, emotion_data.ECG,
                                               color=color_dict[emotion], label=emotion, axis=axs[0])
            median_cleaned = emotion_data.groupby("FILE_NAME", group_keys=False).apply(
                lambda x: apply_rollingmedian(x, 'ECG')).fillna(method="bfill").fillna(method="ffill")
            POPANEFigureGenerator.plot_signals(median_cleaned.time_offset, median_cleaned.ECG,
                                               color=color_dict[emotion], label=emotion, axis=axs[1])
        fig.legend(handles=[Line2D([0], [0], color=color_dict[emotion], lw=2, label=emotion) for emotion in
                            subject_data["EMOTION"].unique()],
                   bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        axs[0].set_title(f"Subject {subject_id} - Raw ECG Signals")
        axs[1].set_title(f"Subject {subject_id} - Median Filtered ECG Signals")
        axs[1].set_xlabel("Time Offset (s)")
        plt.tight_layout()
        plt.show()


def create_aggregated_features(dataset, resample_window='0.1s', batch_size=10):
    if isinstance(dataset, pl.DataFrame):
        dataset = dataset.to_pandas(use_pyarrow_extension_array=True)

    all_files = dataset["File_Name"].unique()
    all_resampled_data = []

    if len(all_files) == 0:
        return pd.DataFrame()

    for batch_idx in range(0, len(all_files), batch_size):
        batch_files = all_files[batch_idx:batch_idx + batch_size]
        print(
            f"  Batch {batch_idx // batch_size + 1}/{(len(all_files) - 1) // batch_size + 1}: Files {batch_idx + 1}-{min(batch_idx + batch_size, len(all_files))}")

        batch_data = []
        for file_name in batch_files:
            file_data = dataset[dataset["File_Name"] == file_name].copy()
            try:
                file_data = file_data.set_index(
                    pd.to_datetime(file_data['timestamp'], unit='s'))
            except Exception:
                file_data = file_data.set_index(
                    pd.to_datetime(file_data['timestamp']))
            for t, w in file_data.resample(resample_window):
                if len(w) == 0:
                    continue

                frame = {}
                for signal in ['ECG', 'EDA', 'SBP', 'DBP']:
                    if signal in w.columns and w[signal].notna().any():
                        signal_values = w[signal].dropna().values
                        frame[f'{signal}_mean'] = w[signal].mean()
                        frame[f'{signal}_median'] = w[signal].median()
                        frame[f'{signal}_std'] = w[signal].std()
                        frame[f'{signal}_min'] = w[signal].min()
                        frame[f'{signal}_max'] = w[signal].max()

                        if len(signal_values) > 10:
                            try:
                                N = len(signal_values)
                                if len(w) > 1:
                                    T = (w.index[-1] - w.index[0]
                                         ).total_seconds() / (len(w) - 1)
                                else:
                                    T = 0.001
                                yf = fft(signal_values)
                                xf = fftfreq(N, T)[:N // 2]

                                dominant_freq = get_dominant_frequency(xf, yf)
                                frame[f'{signal}_dominant_freq'] = dominant_freq

                                signal_energy = calculate_signal_energy(yf)
                                frame[f'{signal}_signal_energy'] = signal_energy

                                del yf, xf
                            except Exception:
                                frame[f'{signal}_dominant_freq'] = np.nan
                                frame[f'{signal}_signal_energy'] = np.nan
                        else:
                            frame[f'{signal}_dominant_freq'] = np.nan
                            frame[f'{signal}_signal_energy'] = np.nan
                    else:
                        frame[f'{signal}_mean'] = np.nan
                        frame[f'{signal}_median'] = np.nan
                        frame[f'{signal}_std'] = np.nan
                        frame[f'{signal}_min'] = np.nan
                        frame[f'{signal}_max'] = np.nan
                        frame[f'{signal}_dominant_freq'] = np.nan
                        frame[f'{signal}_signal_energy'] = np.nan
                frame['Emotion'] = w['Emotion'].iloc[0] if 'Emotion' in w.columns and len(
                    w['Emotion'].dropna()) > 0 else None
                frame['Subject_ID'] = w['Subject_ID'].iloc[0] if 'Subject_ID' in w.columns and len(
                    w['Subject_ID'].dropna()) > 0 else None
                frame['File_Name'] = file_name
                frame['timestamp'] = t.timestamp()
                frame['Study_name'] = w['Study_name'].iloc[0] if 'Study_name' in w.columns and len(
                    w['Study_name'].dropna()) > 0 else None

                batch_data.append(frame)

        if batch_data:
            all_resampled_data.append(pd.DataFrame(batch_data))
        del batch_files, batch_data
        gc.collect()

    if len(all_resampled_data) == 0:
        return pd.DataFrame()

    result = pd.concat(all_resampled_data, ignore_index=True)
    del all_resampled_data
    gc.collect()
    return result
