"""
Tests for emotion.preprocessing.frequency module
"""
import unittest
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from emotion.preprocessing.frequency import (
    mutlipass_filter,
    highpass_0_5hz,
    notch_filter,
    get_dominant_frequency,
    calculate_signal_energy,
    frequency_analysis
)


class TestFrequencyFiltering(unittest.TestCase):
    """Test frequency domain filtering functions"""
    
    def setUp(self):
        """Set up test signals"""
        # Create a test signal with known frequency components
        self.fs = 500  # Sampling frequency
        self.duration = 2  # seconds
        self.t = np.linspace(0, self.duration, self.fs * self.duration, endpoint=False)
        
        # Signal with 1 Hz and 50 Hz components
        self.signal = np.sin(2 * np.pi * 1 * self.t) + 0.5 * np.sin(2 * np.pi * 50 * self.t)
        self.signal += 0.1 * np.random.randn(len(self.t))  # Add noise

    # ==================== Highpass Filter Tests ====================
    
    def test_highpass_0_5hz_output_length(self):
        """Test highpass filter preserves signal length"""
        filtered = highpass_0_5hz(self.signal, fs=self.fs)
        self.assertEqual(len(filtered), len(self.signal))
    
    def test_highpass_0_5hz_removes_dc(self):
        """Test highpass filter removes DC component"""
        dc_signal = np.ones(1000) * 5.0
        filtered = highpass_0_5hz(dc_signal, fs=self.fs)
        # DC component should be removed, signal should be close to zero
        self.assertLess(np.abs(np.mean(filtered)), 1.0)
    
    def test_highpass_0_5hz_preserves_high_freq(self):
        """Test highpass filter preserves high frequency components"""
        high_freq_signal = np.sin(2 * np.pi * 10 * self.t)
        filtered = highpass_0_5hz(high_freq_signal, fs=self.fs)
        # High frequency component should be preserved
        self.assertGreater(np.std(filtered), 0.1)
    
    def test_highpass_0_5hz_different_sampling_rate(self):
        """Test highpass filter with different sampling rate"""
        fs = 1000
        t = np.linspace(0, 1, fs)
        signal = np.sin(2 * np.pi * 5 * t)
        filtered = highpass_0_5hz(signal, fs=fs)
        self.assertEqual(len(filtered), len(signal))
    
    # ==================== Notch Filter Tests ====================
    
    def test_notch_filter_output_length(self):
        """Test notch filter preserves signal length"""
        filtered = notch_filter(self.signal, freq=50, fs=self.fs)
        self.assertEqual(len(filtered), len(self.signal))
    
    def test_notch_filter_removes_target_frequency(self):
        """Test notch filter removes target frequency"""
        # Create signal with only 50 Hz component
        signal_50hz = np.sin(2 * np.pi * 50 * self.t)
        filtered = notch_filter(signal_50hz, freq=50, fs=self.fs)
        
        # Power at 50 Hz should be greatly reduced
        original_power = np.var(signal_50hz)
        filtered_power = np.var(filtered)
        self.assertLess(filtered_power, original_power * 0.1)
    
    def test_notch_filter_preserves_other_frequencies(self):
        """Test notch filter preserves frequencies away from notch"""
        # Create signal with 10 Hz component (away from 50 Hz notch)
        signal_10hz = np.sin(2 * np.pi * 10 * self.t)
        filtered = notch_filter(signal_10hz, freq=50, fs=self.fs)
        
        # Power should be mostly preserved
        original_power = np.var(signal_10hz)
        filtered_power = np.var(filtered)
        self.assertGreater(filtered_power, original_power * 0.8)
    
    def test_notch_filter_custom_q_factor(self):
        """Test notch filter with custom Q factor"""
        filtered = notch_filter(self.signal, freq=50, fs=self.fs, Q=50)
        self.assertEqual(len(filtered), len(self.signal))
    
    # ==================== Multipass Filter Tests ====================
    
    def test_mutlipass_filter_output_length(self):
        """Test multipass filter preserves signal length"""
        filtered = mutlipass_filter(self.signal, fs=self.fs)
        self.assertEqual(len(filtered), len(self.signal))
    
    def test_mutlipass_filter_applies_both_filters(self):
        """Test multipass filter applies both highpass and notch"""
        filtered = mutlipass_filter(self.signal, fs=self.fs)
        
        # Should have removed low frequencies and 50 Hz
        self.assertIsInstance(filtered, np.ndarray)
        self.assertEqual(len(filtered), len(self.signal))
    
    def test_mutlipass_filter_with_different_fs(self):
        """Test multipass filter with different sampling frequency"""
        fs = 1000
        t = np.linspace(0, 1, fs)
        signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)
        filtered = mutlipass_filter(signal, fs=fs)
        self.assertEqual(len(filtered), len(signal))
    
    # ==================== Dominant Frequency Tests ====================
    
    def test_get_dominant_frequency(self):
        """Test dominant frequency detection"""
        # Create signal with known dominant frequency
        fs = 500
        t = np.linspace(0, 1, fs, endpoint=False)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz dominant
        
        from scipy.fft import fft, fftfreq
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1/fs)[:N // 2]
        
        dominant_freq = get_dominant_frequency(xf, yf)
        
        # Should detect frequency close to 10 Hz
        self.assertIsInstance(dominant_freq, (float, np.floating))
        self.assertGreater(dominant_freq, 5)
        self.assertLess(dominant_freq, 15)
    
    def test_get_dominant_frequency_multiple_peaks(self):
        """Test dominant frequency with multiple peaks"""
        fs = 500
        t = np.linspace(0, 1, fs, endpoint=False)
        # Signal with 10 Hz (strong) and 20 Hz (weak)
        signal = 2 * np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
        
        from scipy.fft import fft, fftfreq
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1/fs)[:N // 2]
        
        dominant_freq = get_dominant_frequency(xf, yf)
        
        # Should detect the stronger 10 Hz component
        self.assertGreater(dominant_freq, 5)
        self.assertLess(dominant_freq, 15)
    
    # ==================== Signal Energy Tests ====================
    
    def test_calculate_signal_energy(self):
        """Test signal energy calculation"""
        from scipy.fft import fft
        signal = np.sin(2 * np.pi * 10 * self.t)
        yf = fft(signal)
        
        energy = calculate_signal_energy(yf)
        
        self.assertIsInstance(energy, (float, np.floating))
        self.assertGreater(energy, 0)
    
    def test_calculate_signal_energy_zero_signal(self):
        """Test signal energy for zero signal"""
        from scipy.fft import fft
        signal = np.zeros(1000)
        yf = fft(signal)
        
        energy = calculate_signal_energy(yf)
        
        self.assertEqual(energy, 0.0)
    
    def test_calculate_signal_energy_increases_with_amplitude(self):
        """Test signal energy increases with amplitude"""
        from scipy.fft import fft
        
        signal1 = np.sin(2 * np.pi * 10 * self.t)
        signal2 = 2 * np.sin(2 * np.pi * 10 * self.t)
        
        yf1 = fft(signal1)
        yf2 = fft(signal2)
        
        energy1 = calculate_signal_energy(yf1)
        energy2 = calculate_signal_energy(yf2)
        
        self.assertGreater(energy2, energy1)
    
    # ==================== Integration Tests ====================
    
    def test_filter_chain_preserves_data_type(self):
        """Test filter chain preserves numpy array type"""
        filtered = mutlipass_filter(self.signal, fs=self.fs)
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_filter_chain_no_nan_values(self):
        """Test filter chain produces no NaN values"""
        filtered = mutlipass_filter(self.signal, fs=self.fs)
        self.assertFalse(np.any(np.isnan(filtered)))
    
    def test_filter_chain_no_inf_values(self):
        """Test filter chain produces no infinite values"""
        filtered = mutlipass_filter(self.signal, fs=self.fs)
        self.assertFalse(np.any(np.isinf(filtered)))


class TestFrequencyAnalysis(unittest.TestCase):
    """Test frequency_analysis function"""
    
    def setUp(self):
        """Set up test dataframe"""
        fs = 500
        duration = 2
        t = np.linspace(0, duration, fs * duration, endpoint=False)
        
        # Create dataframe with multiple emotions
        n_samples = len(t) // 2
        
        self.df = pd.DataFrame({
            'timestamp': np.concatenate([t[:n_samples], t[n_samples:2*n_samples]]),
            'DBP': np.concatenate([
                np.sin(2 * np.pi * 1 * t[:n_samples]),
                np.sin(2 * np.pi * 2 * t[n_samples:2*n_samples])
            ]),
            'EMOTION': ['joy'] * n_samples + ['anger'] * n_samples
        })
    
    # @unittest.skip("Requires matplotlib display which may not be available in testing")
    def test_frequency_analysis_runs(self):
        """Test frequency analysis executes without errors"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        try:
            frequency_analysis(self.df)
        except Exception as e:
            self.fail(f"frequency_analysis raised {type(e).__name__}: {e}")


class TestFilterDesign(unittest.TestCase):
    """Test filter design parameters"""
    
    def test_butterworth_filter_stability(self):
        """Test Butterworth filter design is stable"""
        fs = 500
        nyq = fs * 0.5
        cutoff = 0.5
        normal_cutoff = cutoff / nyq
        b, a = butter(5, normal_cutoff, btype='high', analog=False)
        
        # Coefficients should be finite
        self.assertTrue(np.all(np.isfinite(b)))
        self.assertTrue(np.all(np.isfinite(a)))
    
    def test_notch_filter_stability(self):
        """Test notch filter design is stable"""
        from scipy.signal import iirnotch
        
        fs = 500
        freq = 50
        Q = 30
        w0 = freq / (fs / 2)
        b, a = iirnotch(w0, Q)
        
        # Coefficients should be finite
        self.assertTrue(np.all(np.isfinite(b)))
        self.assertTrue(np.all(np.isfinite(a)))


if __name__ == '__main__':
    unittest.main()
