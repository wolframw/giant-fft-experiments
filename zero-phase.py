import argparse
from scipy.io import wavfile
from scipy.fft import rfft, irfft
import numpy as np
from pathlib import Path
import util

p = argparse.ArgumentParser(prog="stn", description="Zero-Phase Resynthesis via Giant FFT")
p.add_argument("--in", required=True, dest="input_file", help="the input file")
p.add_argument("--padding-factor", type=float, default=2, help="amount by which to zero pad, in multiples of the input length (default: %(default)s)")
p.add_argument("--epsilon", type=float, default=0.05, required=False, help="epsilon parameter for gain compensation")
p.add_argument("--out-dir", default=".", required=False, help="the output directory (default: %(default)s)")
p.add_argument("--no-fade", action="store_true" , required=False, help="if specified, does not fade-in and fade-out transients at the beginning and end of the zero-phase signal")
p.add_argument("--no-normalize", action="store_true" , required=False, help="if specified, does not normalize the zero-phase signal to -1/+1 before exporting")
args = p.parse_args()

fs, signal = util.load_wav(args.input_file)
signal = util.pad_odd(signal, args.padding_factor)

spec = rfft(signal, len(signal))
zero_signal_stereo1 = irfft(np.abs(np.real(spec)), len(signal))
zero_signal_stereo2 = irfft(np.abs(np.imag(spec)), len(signal))
zero_signal_mono    = irfft(np.abs(spec), len(signal))
zero_signal_mono   *= util.zero_phase_gain_comp(len(zero_signal_mono), args.epsilon)

if not args.no_fade:
    envelope = util.sine_envelope(len(zero_signal_mono), util.calc_transient_length(zero_signal_mono))
    zero_signal_stereo1 *= envelope
    zero_signal_stereo2 *= envelope
    zero_signal_mono    *= envelope

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
input_file_base = Path(args.input_file).stem
util.write_wav(f"{args.out_dir}/{input_file_base}-mono.wav", fs, zero_signal_mono, normalize=not args.no_normalize)
util.write_wav(f"{args.out_dir}/{input_file_base}-stereo.wav", fs, np.column_stack((zero_signal_stereo1, zero_signal_stereo2)), normalize=not args.no_normalize)
