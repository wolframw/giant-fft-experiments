import argparse
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from scipy.ndimage import median_filter
import numpy as np
from pathlib import Path
import util

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

p = argparse.ArgumentParser(prog="stn", description="Sine-Transient-Noise Decomposition", epilog="To tune the STN algorithm's parameters, you'll have to edit the script because there are so many.")
p.add_argument("--in", required=True, dest="input_file", help="the input file")
p.add_argument("--out-dir", default=".", required=False, help="the output directory (default: %(default)s)")
args = p.parse_args()

fs, in_signal = util.load_wav(args.input_file)

# first step
beta_u1    = 0.8
beta_l1    = 0.7
frame_len1 = 8192
hop1       = int(frame_len1 * 0.25)
freq_bins1 = 93
time_bins1 = 4
stft1      = ShortTimeFFT(win=hann(frame_len1), hop=hop1, fs=fs)

in_stft   = stft1.stft(in_signal)
mag_spec1 = np.abs(in_stft)
xh1       = median_filter(mag_spec1, size=(1,time_bins1), mode="constant", cval=0)
xv1       = median_filter(mag_spec1, size=(freq_bins1,1), mode="constant", cval=0)
rs1       = xh1/(xh1 + xv1)
rt1       = 1 - rs1

sin_mask1       = mask(rs1, beta_l1, beta_u1)
transient_mask1 = mask(rt1, beta_l1, beta_u1)
noise_mask1     = 1 - sin_mask1 - transient_mask1

sin_spec1       = sin_mask1 * in_stft
transient_spec1 = transient_mask1 * in_stft
noise_spec1     = noise_mask1 * in_stft

sin_signal = stft1.istft(sin_spec1)

# second step
beta_u2    = 0.85
beta_l2    = 0.75
frame_len2 = 512
hop2       = int(frame_len2 * 0.25)
freq_bins2 = 6
time_bins2 = 69
stft2      = ShortTimeFFT(win=hann(frame_len2), hop=hop2, fs=fs)

res_spec   = transient_spec1 + noise_spec1
res_signal = stft1.istft(res_spec)
res_stft   = stft2.stft(res_signal)
mag_spec2  = np.abs(res_stft)
xh2        = median_filter(mag_spec2, size=(1,time_bins2), mode="constant", cval=0)
xv2        = median_filter(mag_spec2, size=(freq_bins2,1), mode="constant", cval=0)
rs2        = xh2/(xh2 + xv2)
rt2        = 1 - rs2

sin_mask2       = mask(rs2, beta_l2, beta_u2)
transient_mask2 = mask(rt2, beta_l2, beta_u2)
noise_mask2     = 1 - sin_mask2 - transient_mask2

sin_spec2       = sin_mask2 * res_stft 
transient_spec2 = transient_mask2 * res_stft 
noise_spec2     = noise_mask2 * res_stft

transient_signal = stft2.istft(transient_spec2)
noise_signal     = stft2.istft(noise_spec2)

# output
Path(args.out_dir).mkdir(parents=True, exist_ok=True)
input_file_base = Path(args.input_file).stem
util.write_wav(f"{args.out_dir}/{input_file_base}-sin.wav", fs, sin_signal)
util.write_wav(f"{args.out_dir}/{input_file_base}-transient.wav", fs, transient_signal)
util.write_wav(f"{args.out_dir}/{input_file_base}-noise.wav", fs, noise_signal)
