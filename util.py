from scipy.io import wavfile
import numpy as np

def load_wav(path, normalize=True):
    fs, signal = wavfile.read(path)
    signal = signal.astype(np.float32)

    if normalize:
        m = max(np.abs(signal))
        if m > 0:
            signal /= m

    return fs, signal

def write_wav(path, fs, signal, normalize=True):
    signal = np.array(signal, dtype=np.float32)

    if normalize:
        peak = np.max(np.abs(signal), axis=0)
        if signal.ndim == 1:
            if peak == 0:
                peak = 1
        else:
            peak[peak == 0] = 1
        signal /= peak
    wavfile.write(path, fs, signal)

def pad_odd(signal, factor):
    factor = max(1, factor)
    pad_length = int((factor - 1) * len(signal))
    if pad_length == 0:
        return signal
    if (pad_length + len(signal)) % 2 == 0:
        pad_length = pad_length - 1
    return np.pad(signal, (0, pad_length))

def sine_envelope(length, fade_length):
    w = np.ones(max(0, length))

    if fade_length > 0:
        x = np.arange(fade_length)
        w[:fade_length]  = np.sin(np.pi * x * 0.5 / fade_length)
        w[-fade_length:] = np.cos(np.pi * x * 0.5 / fade_length)

    return w

def calc_transient_length(signal, chunk_length=500, hop=250):
    offset = 0
    median_sample = np.median(np.abs(signal))

    while offset + chunk_length < len(signal):
        chunk   = signal[offset:offset + chunk_length]
        offset += hop

        if np.median(np.abs(chunk)) < median_sample:
            return offset

    return 1

def zero_phase_gain_comp(total_length, epsilon):
    ramp_length = np.int64(np.ceil(total_length / 2))
    ramp = np.linspace(1, 0, ramp_length)
    g = 1 / (np.sqrt(ramp) + epsilon)
    if total_length % 2 == 0:
        g = np.concatenate([g, np.flip(g)])
    else:
        g = np.concatenate([g, np.flip(g[:-1])])
    return g
