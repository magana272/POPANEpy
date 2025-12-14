from typing import cast

import matplotlib.pyplot as plt
# from numpy.typing import Unknown
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal import iirnotch


##########################################################
# Preprocessing - Frequency Domain Filtering

def mutlipass_filter(data, fs=500):
    filtered = highpass_0_5hz(data, fs)
    filtered = notch_filter(filtered, freq=50, fs=fs)
    return filtered


def highpass_0_5hz(data, fs=500):
    nyq = fs * 0.5
    cutoff = 0.5
    normal_cutoff = cutoff / nyq
    res = butter(5, normal_cutoff, btype='high', analog=False)
    if res is not None and len(res) == 2:
        b, a = res
    else:
        raise ValueError("Butterworth filter design failed.")
    return filtfilt(b, a, data)


def notch_filter(data, freq, fs=500, Q=30):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data)


def frequency_analysis(subject: pd.DataFrame):
    dom_freq_singal_energy = {emotion: {}
                              for emotion in subject["Emotion"].unique()}
    colors = sns.color_palette("husl", subject["Emotion"].nunique())
    color_dict = {emotion: colors[i]
                  for i, emotion in enumerate(subject["Emotion"].unique())}
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, emotion in enumerate(subject["Emotion"].unique()):
        emotion_data = subject[subject["Emotion"] == emotion]
        emotion_data = emotion_data['DBP'].values
        N = len(emotion_data)
        T = subject.timestamp[1] - subject.timestamp[0]  # sampling interval
        yf = fft(emotion_data)
        xf = fftfreq(N, T)[:N // 2]
        dominant_freq = get_dominant_frequency(xf, yf)
        # dom_freq_singal_energy[emotion]['dominant_frequency'] = dominant_freq
        signal_energy = np.log2(calculate_signal_energy(yf))
        # dom_freq_singal_energy[emotion]['signal_energy'] = signal_energy
        print(f"Dominant Frequency for {emotion}: {dominant_freq} Hz")
        print(f"Signal Energy for {emotion}: {signal_energy}")
        ax.plot(xf, (2.0 / N * np.abs(cast(np.ndarray,
                                           yf[0:N // 2]))), label=emotion, color=color_dict[emotion])
        ax.set_title('Frequency Spectrum of {}'.format(emotion))
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Normalized Magnitude')
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 8)
        ax.legend()
    plt.show()


def get_dominant_frequency(xf, yf):
    N = len(yf)
    peaks, _ = find_peaks(np.abs(yf[:N // 2]), height=0)
    dominant_freq = xf[peaks[np.argmax(np.abs(yf[peaks]))]]
    return dominant_freq


def calculate_signal_energy(yf):
    """
    Signal energy gives an overall measure of the intensity of variations in the accelerometer data.
    It's calculated as the sum of the squared magnitudes of the FFT results.

    Parameters:
    - yf: FFT result array

    Returns:
    - signal_energy: The calculated signal energy
    """
    return np.sum(np.abs(yf) ** 2) / len(yf)


class FrequencyDomainFeatures:
    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
        self.base_freqs = [
            ('ECG', (0.5, 40))]

    def compute_fft(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N = len(signal)
        yf = cast(np.ndarray, fft(signal))
        xf = cast(np.ndarray, fftfreq(N, 1 / self.sampling_rate))
        return (xf, yf)


class FrequencyDomainFeatureExtractor:
    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
        self.freq_domain = FrequencyDomainFeatures(sampling_rate)

    def extract_features(self, signal: np.ndarray) -> dict[str, float]:
        xf, yf = self.freq_domain.compute_fft(signal)
        features = {}
        features['dominant_frequency'] = get_dominant_frequency(xf, yf)
        features['signal_energy'] = calculate_signal_energy(yf)
        return features
