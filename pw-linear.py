import argparse
from scipy.io import wavfile
from scipy.fft import rfft, irfft
from scipy.stats import linregress
import numpy as np
from pathlib import Path
import util

p = argparse.ArgumentParser(prog="pw-linear", description="Piecewise Linear-Phase Resynthesis via Giant FFT and Linear Regression")
p.add_argument("--in", required=True, dest="input_file", help="the input file")
p.add_argument("--padding-factor", type=float, default=2, help="amount by which to zero pad, in multiples of the input length (default: %(default)s)")
p.add_argument("--specframes", type=int, default=1, help="number of spectral frames (default: %(default)s)")
p.add_argument("--out-dir", default=".", help="the output directory (default: %(default)s)")
args = p.parse_args()

fs, signal = util.load_wav(args.input_file)
signal = util.pad_odd(signal, args.padding_factor)

fft_length = len(signal)
spec       = rfft(signal, fft_length)
phase      = np.unwrap(np.angle(spec))
frame_bins = len(spec) // args.specframes
pw_phase   = np.empty_like(phase)

print(f"{len(spec)} bins in total")
print(f"{args.specframes} frames, covering {frame_bins} bins each (~ {frame_bins / len(spec) * 100}%))")

for f in range(args.specframes):
    start = f * frame_bins
    end = min(len(spec), start + frame_bins)

    k = np.arange(start, end)
    frame = phase[start:end]
    reg = linregress(k, frame)
    frame = reg.slope * k + reg.intercept
    pw_phase[start:end] = frame

pw_spec = np.abs(spec) * np.exp(1j * pw_phase)
pw_signal = irfft(pw_spec, fft_length)

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
util.write_wav(f"{args.out_dir}/{Path(args.input_file).stem}-{args.specframes}.wav", fs, pw_signal, normalize=False)
