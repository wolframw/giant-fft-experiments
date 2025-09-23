from scipy.io import wavfile
from scipy.fft import rfft, irfft
from scipy.ndimage import median_filter
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os

def frame_signal(x, frame_len, hop):
    x = np.asarray(x)
    n = x.shape[0]
    n_frames = 1 + (n - frame_len) // hop if n >= frame_len else 1
    pad = max(0, frame_len + (n_frames - 1) * hop - n)
    if pad:
        x = np.pad(x, (0, pad))
    n = x.shape[0]
    n_frames = 1 + (n - frame_len) // hop
    stride = x.strides[0]
    return as_strided(x, shape=(n_frames, frame_len), strides=(hop*stride, stride))

def stft(x, frame_len, hop, window):
    frames = frame_signal(x, frame_len, hop)
    win_frames = frames * window[None, :]
    return rfft(win_frames, axis=1)

def istft(x, frame_len, hop, window):
    n_frames, n_bins = x.shape
    y_len = frame_len + (n_frames - 1) * hop
    y = np.zeros(y_len, dtype=np.float32)
    wsum = np.zeros(y_len, dtype=np.float32)

    for i in range(n_frames):
        frame = irfft(x[i], n=frame_len).astype(np.float32)
        start = i * hop
        y[start:start+frame_len] += frame * window
        wsum[start:start+frame_len] += window**2

    nz = wsum > 1e-12
    y[nz] /= wsum[nz]
    return y

def median(mag_spec, filter_length, axis):
    size = [1] * mag_spec.ndim
    size[axis] = filter_length
    return median_filter(mag_spec, size=tuple(size), mode="constant", cval=0)

def mask(a, beta_l, beta_u):
    a = np.asarray(a)
    return np.where(
        a >= beta_u,
        1.0,
        np.where(
            (a >= beta_l) & (a < beta_u),
            np.sin(np.pi/2 * (a - beta_l) / (beta_u - beta_l))**2,
            0.0
        )
    )

input_file = "techno-drums.wav"

sr, data = wavfile.read(f"in/{input_file}")
peak = np.max([np.abs(np.iinfo(data.dtype).min), np.abs(np.iinfo(data.dtype).max)])
data = data.astype(np.float32) / np.abs(peak)

beta_u1 = 0.8
beta_l1 = 0.7
beta_u2 = 0.85
beta_l2 = 0.75
frame_len1 = 8192
hop1       = int(frame_len1 * 0.75)
freq_bins1 = 93
time_bins1 = 4
frame_len2 = 512
hop2       = int(frame_len2 * 0.75)
freq_bins2 = 6
time_bins2 = 69
window1 = np.hanning(frame_len1).astype(np.float32)
window2 = np.hanning(frame_len2).astype(np.float32)
x_stft = stft(data, frame_len1, hop1, window1)
mag_spec1 = np.abs(x_stft)
xh1 = median(mag_spec1, 5, 0)
xv1 = median(mag_spec1, 9, 1)

# first step

rs1 = xh1/(xh1 + xv1)
rt1 = 1 - rs1
rn1 = 1 - np.sqrt(np.abs(rs1 - rt1))

sin_mask1       = mask(rs1, beta_l1, beta_u1)
transient_mask1 = mask(rt1, beta_l1, beta_u1)
noise_mask1     = 1 - sin_mask1 - transient_mask1

sin_spec1       = sin_mask1 * x_stft
transient_spec1 = transient_mask1 * x_stft
noise_spec1     = noise_mask1 * x_stft

sin_signal = istft(sin_spec1, frame_len1, hop1, window1)

# second step

res_spec = transient_spec1 + noise_spec1
res_signal = istft(res_spec, frame_len1, hop1, window1)
res_stft = stft(res_signal, frame_len2, hop2, window2)
mag_spec2 = np.abs(res_stft)
xh2 = median(mag_spec2, 5, 0)
xv2 = median(mag_spec2, 9, 1)

rs2 = xh2/(xh2 + xv2)
rt2 = 1 - rs2

sin_mask2        = mask(rs2, beta_l2, beta_u2)
transient_mask2  = mask(rt2, beta_l2, beta_u2)
noise_mask2      = 1 - sin_mask2 - transient_mask2

sin_spec2       = sin_mask2 * res_stft 
transient_spec2 = transient_mask2 * res_stft 
noise_spec2     = noise_mask2 * res_stft

transient_signal = istft(transient_spec2, frame_len2, hop2, window2)
noise_signal     = istft(noise_spec2, frame_len2, hop2, window2)

filename = os.path.basename(input_file)
wavfile.write(f"out/{filename}-sin.wav", sr, sin_signal)
wavfile.write(f"out/{filename}-transient.wav", sr, transient_signal)
wavfile.write(f"out/{filename}-noise.wav", sr, noise_signal)
