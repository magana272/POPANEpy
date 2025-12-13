"""
Module for generating figures for POPANE emotion study data.

Includes functions to plot physiological signals over time,
differentiated by emotion.
Functions:
- plot_signals: Plot a generic signal over time.
- create_figure_for_subject: Plot physiological 
signals for a given subject,differentiating by emotion
- create_figure_one_per_study: Create figures for each subject in a study.
"""
from typing import Any, TypeAlias
from polars import col
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import numpy as np
import pandas as pd
_Shape: TypeAlias = tuple[int, ...]
_AnyShape: TypeAlias = tuple[Any, ...]


TOATALFEATURELIST = ['ECG', 'EDA', 'SBP', 'DBP',
                     'respiration', 'temp', 'CO', 'TPR',
                     'affect', 'dz', 'z0', 'dzdt']


class POPANEFigureGenerator:
    """
    A class to generate figures for POPANE emotion study data.
    """
    feature_plot_functions = {}

    def __init__(self):
        self._set_up_figure_styles()

    def _set_up_figure_styles(self):
        """
        Set up the default styles for the figures.
        """
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.titlesize': 16,
            'axes.titleweight': 'bold',
            'axes.labelsize': 14,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'legend.fontsize': 12,
            'font.size': 12
        })
        self.feature_plot_functions = {
            'ECG': {'plotfn': self.plot_signals,
                    'title': 'ECG Signal',
                    'y_label': 'ECG\n(mV)',
                    'color': 'blue'},
            'EDA': {'plotfn': self.plot_signals,
                    'title': 'EDA Signal',
                    'y_label': 'EDA\n(µS)',
                    'color': 'green'},
            'SBP': {'plotfn': self.plot_signals,
                    'title': 'SBP Signal',
                    'y_label': 'SBP\n(mmHg)',
                    'color': 'red'},
            'DBP': {'plotfn': self.plot_signals,
                    'title': 'DBP Signal',
                    'y_label': 'DBP\n(mmHg)',
                    'color': 'purple'},
            'respiration': {'plotfn': self.plot_signals,
                            'title': 'Respiration Signal',
                            'y_label': 'Res.\n(breaths/min)',
                            'color': 'orange'},
            'temp': {'plotfn': self.plot_signals,
                     'title': 'Temperature Signal',
                     'y_label': 'Temp\n(°C)',
                     'color': 'brown'},
            'CO': {'plotfn': self.plot_signals,
                   'title': 'Cardiac Output Signal',
                   'y_label': 'CO\n(l/min)',
                   'color': 'pink'},
            'TPR': {'plotfn': self.plot_signals,
                    'title': 'Total Peripheral Resistance',
                    'y_label': 'TPR\n(mmHg*min/l)',
                    'color': 'cyan'},
            'affect': {'plotfn': self.plot_signals,
                       'title': 'Affect',
                       'y_label': 'Affect',
                       'color': 'magenta'},
            'dz': {'plotfn': self.plot_signals,
                   'title': 'dz Signal',
                   'y_label': 'dz\n(ohm)',
                   'color': 'yellow'},
            'z0': {'plotfn': self.plot_signals,
                   'title': 'z0 Signal',
                   'y_label': 'z0\n(ohm)',
                   'color': 'gray'},
            'dzdt': {'plotfn': self.plot_signals,
                     'title': 'dzdt Signal',
                     'y_label': 'dzdt\n(ohm/s)',
                     'color': 'black'}
        }

    @staticmethod
    def plot_signals(t: np.ndarray, signal: np.ndarray,
                     **kwargs) -> matplotlib.axes.Axes | Any:
        """
        Plot a generic
        signal over time.
        """
        title = kwargs.get('title', 'Signal')
        y_label = kwargs.get('y_label', 'Value')
        color = kwargs.get('color', 'green')
        label = kwargs.get('label', None)
        axis = kwargs.get('axis', None)
        legend = kwargs.get('legend', True)
        linestyle = kwargs.get('linestyle', '-')
        if axis is None:
            _, axis = plt.subplots(figsize=(10, 4))
        axis.plot(t, signal, color=color, label=label,
                  linestyle=linestyle, **kwargs)
        axis.set_title(title, x=0.85, y=1.0, pad=-14,
                       fontsize=15, fontweight='bold')
        axis.set_ylabel(y_label, fontsize=12)
        axis.grid(True, alpha=0.3)
        if (legend and label is not None):
            axis.legend(loc='upper right', fontsize=9, framealpha=0.9)
        return axis

    @staticmethod
    def create_figure_for_subject(df: pd.DataFrame, subject_id: int,
                                  start_end=(0, 10),
                                  figsize=(16, 12)) -> matplotlib.figure.Figure | None:
        """ 
        Plot physiological signals for a given subject, differentiating by emotion.
        """
        subject_data = df[df['Subject_ID'] == subject_id].copy()
        if len(subject_data) == 0:
            print(f"No data found for Subject_ID: {subject_id}")
            return None

        total_feature_list = TOATALFEATURELIST
        features_in_data: list[str] = [feature for feature in subject_data.columns if
                                       feature in total_feature_list]
        unique_recordings = subject_data['File_Name'].unique()
        # emotions = subject_data['Emotion'].unique()
        study = subject_data['Study_name'].iloc[0]
        emotion_colors = {}
        palette = sns.color_palette("husl", n_colors=len(unique_recordings))
        for i, recording in enumerate(unique_recordings):
            emotion = subject_data[subject_data['File_Name']
                                   == recording]['Emotion'].iloc[0]
            emotion_colors[recording] = {
                'color': palette[i], 'emotion': emotion}
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(len(features_in_data), hspace=0)
        axes: Any = gs.subplots(sharex=False, sharey=False)
        if isinstance(axes, matplotlib.axes.Axes):
            axes_dict = {axes: features_in_data[0]}
        else:
            axes_dict: dict[matplotlib.axes.Axes, str] = dict(
                zip(axes, features_in_data))
        for recording in unique_recordings:
            recording_data = subject_data[subject_data['File_Name'] == recording].copy(
            )
            emotion = emotion_colors[recording]['emotion']
            color = emotion_colors[recording]['color']
            recording_data['time_offset'] = recording_data['timestamp'] - \
                recording_data['timestamp'].iloc[0]
            if start_end[0] is not None:
                recording_data = recording_data[recording_data['time_offset']
                                                >= start_end[0]]
            if start_end[1] is not None:
                recording_data = recording_data[recording_data['time_offset']
                                                <= start_end[1]]
            for iax, (ax, feature) in enumerate(axes_dict.items()):
                if feature in recording_data.columns:
                    feature_plot_function = POPANEFigureGenerator.feature_plot_functions[
                        feature]["plotfn"]
                    title = POPANEFigureGenerator.feature_plot_functions[feature]["title"]
                    y_label = POPANEFigureGenerator.feature_plot_functions[feature]["y_label"]
                    if feature_plot_function is not None:
                        feature_plot_function(
                            recording_data['time_offset'],
                            recording_data[feature],
                            title=title,
                            color=color,
                            y_label=y_label,
                            label=emotion,
                            axis=ax
                        )
                    if iax == 0:
                        ax.legend(loc='lower right',
                                  fontsize=9, framealpha=0.9)
                    if iax == len(axes_dict) - 1:
                        ax.set_xlabel('Time (seconds)', fontsize=12,
                                      fontweight='bold')
        fig.suptitle(
            f'Physiological Signals for Subject {subject_id} in {study}',
            fontsize=16,
            fontweight='bold')
        plt.tight_layout()
        return fig

    @staticmethod
    def create_figure_one_per_study(study_data_dict: dict[str, pd.DataFrame],
                                    start_end=(0, 10)) -> list[matplotlib.figure.Figure]:
        """
        Docstring for create_figure_one_per_study
        :param study_data_dict: Description
        :type study_data_dict: dict
        """
        figures = []
        for _, (_, subject_data) in enumerate(study_data_dict.items()):
            subject_id = subject_data.Subject_ID.iloc[0]
            fig = POPANEFigureGenerator.create_figure_for_subject(subject_data,
                                                                  subject_id=subject_id, figsize=(
                                                                      10, 10), start_end=start_end)
            if fig is not None:
                figures.append(fig)
        return figures


emotion_figure_generator = POPANEFigureGenerator()
