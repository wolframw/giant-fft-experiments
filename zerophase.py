from scipy.io import wavfile
from scipy.fft import rfft, irfft
import numpy as np;

def sine_window(length, fade):
    fade = max(0, fade)
    w = np.ones(length)

    if fade > 0:
        x = np.arange(fade)
        w[:fade] = np.sin(np.pi * x * 0.5 / fade)
        w[-fade:] = np.cos(np.pi * x * 0.5 / fade)
    
    return w

def gain_comp(total_length, epsilon = 0.05):
    ramp_length = np.int64(total_length / 2)
    ramp = np.linspace(1, 0, ramp_length)
    g = 1 / (np.sqrt(ramp) + epsilon)
    if total_length % 2 == 0:
        g = np.concatenate([g, np.flip(g)])
    else:
        g = np.concatenate([g, np.flip(g[1:])])
    return g

input_file = "techno-drums.wav"

sr, data = wavfile.read(f"in/{input_file}")
peak = np.max([np.abs(np.iinfo(data.dtype).min), np.abs(np.iinfo(data.dtype).max)])
data = data.astype(np.float32) / np.abs(peak)

spec = rfft(data, 2 * len(data) - 1)
mag_spec = np.abs(spec)
zero_phase = irfft(mag_spec)

clip_indices = np.flatnonzero(np.abs(zero_phase[0:int(0.01 * len(data))]) > 1)
last_clip_index = clip_indices[-1] if clip_indices.size else 0
attenuation = np.arcsin(1 / zero_phase[last_clip_index])
fade_length = np.int64(np.ceil(last_clip_index / attenuation * np.pi))
zero_phase = zero_phase * sine_window(len(zero_phase), fade_length) * gain_comp(len(zero_phase))
zero_phase = zero_phase / np.max(np.abs(zero_phase))

output_file = f"out/{input_file}"
wavfile.write(output_file, sr, zero_phase)
