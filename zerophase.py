from scipy.io import wavfile
from scipy.fft import rfft, irfft
import numpy as np;


input_file = "clean-acoustic.wav"

sr, data = wavfile.read(f"in/{input_file}")
peak = np.max([np.abs(np.iinfo(data.dtype).min), np.abs(np.iinfo(data.dtype).max)])
data = data.astype(np.float32) / np.abs(peak)

spec = rfft(data, 2 * len(data) - 1)
mag_spec = np.abs(spec)
zero_phase = irfft(mag_spec)

output_file = f"out/{input_file}"
wavfile.write(output_file, sr, zero_phase)
