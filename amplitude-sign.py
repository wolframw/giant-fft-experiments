from scipy.io import wavfile
from scipy.fft import rfft, irfft
import numpy as np

def nzsign(x):
    x = np.asarray(x)
    result = np.ones_like(x, dtype=int)
    result[np.signbit(x)] = -1
    return result.item() if result.ndim == 0 else result

def sign_real(mag_spec, spec):
    return mag_spec * nzsign(np.real(spec))

def sign_signal(mag_spec, spec, sig):
    return mag_spec * nzsign(sig[:len(mag_spec)])

def sign_sin_sweep(mag_spec, f0, f1):
    return mag_spec * nzsign(np.sin(2*np.pi*np.linspace(f0, f1, len(mag_spec))*np.arange(len(mag_spec))))

def sign_other_signal(mag_spec, sign_sig):
    return mag_spec * nzsign(sign_sig[:len(mag_spec)])

input_file = "clean-acoustic.wav"
sign_file = "techno-drums.wav"

sr, data = wavfile.read(f"in/{input_file}")
sign_sr, sign_data = wavfile.read(f"in/{sign_file}")
shortest = min(len(data), len(sign_data))
data = data[:shortest]
sign_data = sign_data[:shortest]
peak = np.max([np.abs(np.iinfo(data.dtype).min), np.abs(np.iinfo(data.dtype).max)])
data = data.astype(np.float32) / np.abs(peak)
sig_peak = np.max([np.abs(np.iinfo(sign_data.dtype).min), np.abs(np.iinfo(sign_data.dtype).max)])
sign_data = sign_data.astype(np.float32) / np.abs(sig_peak)

spec     = rfft(data, len(data))
mag_spec = np.abs(spec)
sig_sr       = irfft(sign_real(mag_spec, spec), len(data))
sig_sig      = irfft(sign_signal(mag_spec, spec, data), len(data))
sig_f0_9     = irfft(sign_sin_sweep(mag_spec, 0.9, 0.9), len(data))
sig_f0_001   = irfft(sign_sin_sweep(mag_spec, 0.001, 0.001), len(data))
sig_f0_1_1   = irfft(sign_sin_sweep(mag_spec, 0.1, 1), len(data))
sig_f0_2_0_3 = irfft(sign_sin_sweep(mag_spec, 0.2, 0.202), len(data))
sig_othersig = irfft(sign_other_signal(mag_spec, sign_data), len(data))
sign_spec    = rfft(sign_data, len(sign_data))
sig_otherreal = irfft(sign_real(mag_spec, sign_spec), len(data))

wavfile.write(f"out/{input_file}-sign_real.wav", sr, sig_sr)
wavfile.write(f"out/{input_file}-sign_signal.wav", sr, sig_sig)
wavfile.write(f"out/{input_file}-sign_sin_0.09.wav", sr, sig_f0_9)
wavfile.write(f"out/{input_file}-sign_sin_0.001.wav", sr, sig_f0_001)
wavfile.write(f"out/{input_file}-sign_sin_0.1_1.wav", sr, sig_f0_1_1)
wavfile.write(f"out/{input_file}-sign_sin_0.2_0.202.wav", sr, sig_f0_2_0_3)
wavfile.write(f"out/{input_file}-sign_{sign_file}_signal.wav", sr, sig_othersig)
wavfile.write(f"out/{input_file}-sign_{sign_file}_spec.wav", sr, sig_otherreal)
