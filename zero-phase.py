from scipy.io import wavfile
from scipy.fft import rfft, irfft, fftshift
import numpy as np
from pathlib import Path

def sine_window(length, fade):
    fade = max(0, fade)
    w = np.ones(length)

    if fade > 0:
        x = np.arange(fade)
        w[:fade]  = np.sin(np.pi * x * 0.5 / fade)
        w[-fade:] = np.cos(np.pi * x * 0.5 / fade)

    return w

def calc_window_length(signal, target, chunk_length=500, hop=250):
    offset = 0
    while offset + chunk_length < len(signal):
        chunk   = signal[offset:offset + chunk_length]
        offset += hop
        if np.median(np.abs(chunk)) < target:
            return offset
    return 1

def gain_comp(total_length, epsilon = 0.05):
    ramp_length = np.int64(np.ceil(total_length / 2))
    ramp = np.linspace(1, 0, ramp_length)
    g = 1 / (np.sqrt(ramp) + epsilon)
    if total_length % 2 == 0:
        g = np.concatenate([g, np.flip(g)])
    else:
        g = np.concatenate([g, np.flip(g[:-1])])
    return g

def normalize(signal):
    peak = np.max([np.abs(np.iinfo(signal.dtype).min), np.abs(np.iinfo(signal.dtype).max)])
    return signal.astype(np.float32) / np.abs(peak)

input_file = "clean-acoustic.wav"
padding_factor = 2

fs, signal = wavfile.read(f"in/{input_file}")

pad_length = (padding_factor - 1) * len(signal)
if (pad_length + len(signal)) % 2 == 0:
    pad_length = pad_length - 1
padded_signal = np.pad(signal, (0, pad_length))
padded_length = len(padded_signal)

spec = rfft(padded_signal)
zero_signal_stereo1 = irfft(np.abs(np.real(spec)))
zero_signal_stereo2 = irfft(np.abs(np.imag(spec)))
zero_signal_mono    = irfft(np.abs(spec))

comp_env = gain_comp(len(zero_signal_mono))
zero_signal_mono    *= comp_env

median_sample = np.median(np.abs(zero_signal_mono))
window = sine_window(len(zero_signal_mono), calc_window_length(zero_signal_mono, median_sample))

zero_signal_stereo1 *= window
zero_signal_stereo1 /= np.max(np.abs(zero_signal_stereo1))

zero_signal_stereo2 *= window
zero_signal_stereo2 /= np.max(np.abs(zero_signal_stereo2))

zero_signal_mono *= window
zero_signal_mono /= np.max(np.abs(zero_signal_mono))

Path("out/zero-phase").mkdir(parents=True, exist_ok=True)
input_file_base = Path(input_file).stem
wavfile.write(f"out/zero-phase/{input_file_base}-mono.wav"  , fs, zero_signal_mono)
wavfile.write(f"out/zero-phase/{input_file_base}-stereo.wav", fs, np.column_stack([zero_signal_stereo1, zero_signal_stereo2]))
