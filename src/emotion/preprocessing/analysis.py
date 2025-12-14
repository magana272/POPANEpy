from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from emotion import utils
from emotion.studies.dataloader import POPANEDataLoader


def plot_pca(X: np.ndarray, y: np.ndarray) -> None:
    """Plot PCA of the dataset colored by emotion labels.
    Parameters:
    - X: The input data as a numpy array of shape (num_samples, num_features).
    - y: The labels as a numpy array of shape (num_samples,).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    unique_emotions = np.unique(y)
    palette = sns.color_palette("husl", n_colors=len(unique_emotions))

    plt.figure(figsize=(8, 6))
    plt.style.use("dark_background")

    for i, emotion in enumerate(unique_emotions):
        mask = y == emotion
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], c=[
            palette[i]], label=emotion, s=1, alpha=0.7)

    plt.title("PCA of Physiological Data from Study 1", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("PCA_Study1.png")


def get_unique_emotions_for_all_studies(popane_data_loader: POPANEDataLoader):
    all_unique_emotions = set()
    for study_number in range(1, 8):
        unique_emotions = popane_data_loader.get_unique_emotions(study_number)
        all_unique_emotions |= set(unique_emotions)
    return list(set(all_unique_emotions))


def generate_emotion_presence_matrix(all_emotions, popane_data_loader: POPANEDataLoader):
    emotion_matrix = pd.DataFrame(0, index=[
        'Study1', 'Study2', 'Study3', 'Study4', 'Study5', 'Study6', 'Study7'], columns=all_emotions)
    emotions_per_study = [
        popane_data_loader.get_unique_emotions(i + 1) for i in range(7)]
    for i, study_emotions in enumerate(emotions_per_study):
        for emotion in study_emotions:
            s = popane_data_loader.get_study_metadata(i + 1)
            if s is None:
                print(f"Metadata for study {i + 1} is not available.")
                continue
            if s.FILE_NAME is None:
                print(f"FILE_NAME for study {i + 1} is not available.")
                continue
            count = len([x for x in list(zip(s.EMOTION, s.FILE_NAME)) if  x[0] == emotion])
            emotion_matrix.loc[emotion_matrix.index[i],
            emotion] = count
    return emotion_matrix


def generate_heatmap(emotion_matrix):
    plt.style.use("dark_background")
    plt.figure(figsize=(8, 6))
    emotion_matrix = emotion_matrix.T
    flat_map = emotion_matrix.to_numpy().flatten()
    non_zero_values = flat_map[flat_map != 0]
    min_value = np.min(non_zero_values)
    max_value = np.max(non_zero_values)
    sns.heatmap(emotion_matrix,
                annot=emotion_matrix,
                cbar=True,
                cmap="crest",
                fmt="d",
                vmin=min_value,
                vmax=max_value,
                mask=emotion_matrix == 0,
                linewidth=.5)
    plt.title("Emotion Presence Across POPANE Studies", fontsize=20)
    plt.ylabel("Emotions", fontsize=14)
    plt.xlabel("Studies", fontsize=14)
    plt.show()


def generate_study_size_bar_chart(popane_data_loader: POPANEDataLoader):
    datasizes = pd.DataFrame.from_dict(utils.get_size(
        popane_data_loader.data_dir)).T.reset_index()
    datasizes.columns = ["Study", "Size_GB", "File_Count"]
    datasizes["File_Count"] = datasizes["File_Count"].astype(
        int).astype(str) + " Files"
    ax: Axes = sns.barplot(datasizes, x="Study",
                           y="Size_GB", estimator="sum", errorbar=None)
    ax.containers[0] = cast(BarContainer, ax.containers[0])
    _ = ax.bar_label(ax.containers[0],
                     labels=datasizes["File_Count"], fontsize=10)
    ax.set_title("Size of POPANE Study Data Files", fontsize=16)
    ax.set_ylabel("Size (GB)", fontsize=12)
