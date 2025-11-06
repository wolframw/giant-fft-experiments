import argparse
from scipy.io import wavfile
from scipy.fft import rfft, irfft
from scipy.signal import chirp, square, convolve
import numpy as np
from pathlib import Path
import util

def nzsign(x):
    if x.ndim == 0:
        return -1.0 if x < 0 else 1.0
    return np.where(x >= 0, 1.0, -1.0)

def sign_real_spec(mag_spec, spec):
    return mag_spec * nzsign(np.real(spec))

def sign_signal(mag_spec, sig):
    return mag_spec * nzsign(sig[:len(mag_spec)])
    
def sign_chirp(mag_spec, f0, f1, fs):
    return mag_spec * nzsign(chirp(np.arange(len(mag_spec))/fs, f0, len(mag_spec)/fs, f1))

def sign_pwm(mag_spec, f, duty0, duty1, fs):
    return mag_spec * nzsign(square(2 * np.pi * f * np.arange(len(mag_spec)) / fs, np.linspace(max(0, duty0), max(0, duty1), num=len(mag_spec), endpoint=True)))

# def sign_other_s
p = argparse.ArgumentParser(prog="signed-amplitude", description="Signed-Amplitude Resynthesis via Giant FFT")
p.add_argument("--in", required=True, dest="input_file", help="the input file")
p.add_argument("--padding-factor", type=float, default=2, help="amount by which to zero pad, in multiples of the input length (default: %(default)s)")
p.add_argument("--out-dir", default=".", help="the output directory (default: %(default)s)")
sp = p.add_subparsers(dest="mode", required=True, help="where to take the sign from")
sp.add_parser("real_spec", help="the input signal's real part of the spectrum")
sp.add_parser("signal", help="the input signal's samples")
chirp_parser = sp.add_parser("chirp", help="a chirp signal")
chirp_parser.add_argument("--f0", type=float, required=True, help="initial frequency")
chirp_parser.add_argument("--f1", type=float, required=True, help="final frequency")
chirp_parser.add_argument("--postscale", action=argparse.BooleanOptionalAction, required=False, help="apply sign mask after enveloping and scaling")
pwm_parser = sp.add_parser("pwm", help="a PWM square wave")
pwm_parser.add_argument("--f", type=float, required=True, help="frequency")
pwm_parser.add_argument("--duty0", type=float, default=0.5, help="initial duty cycle (between 0.0 and 1.0, default: %(default)s)")
pwm_parser.add_argument("--duty1", type=float, default=0.5, help="final duty cycle (between 0.0 and 1.0, default: %(default)s)")
donor_signal_parser = sp.add_parser("donor_signal", help="a donor signal")
donor_signal_parser.add_argument("--donor", required=True, help="file name of the donor signal")
donor_signal_parser.add_argument("--postscale", action=argparse.BooleanOptionalAction, required=False, help="apply sign mask after enveloping and scaling")
donor_real_spec_parser = sp.add_parser("donor_real_spec", help="the real part of a donor signal's spectrum")
donor_real_spec_parser.add_argument("--donor", required=True, help="file name of the donor signal")
args = p.parse_args()

fs, signal = util.load_wav(args.input_file)

if args.mode == "donor_signal" or args.mode == "donor_real_spec":
    donor_fs, donor_signal = util.load_wav(args.donor)

    if len(donor_signal) > len(signal):
        fitted_signal       = np.pad(signal, (0, len(donor_signal) - len(signal)))
        fitted_donor_signal = donor_signal
    else:
        fitted_signal       = signal
        fitted_donor_signal = np.pad(donor_signal, (0, len(signal) - len(donor_signal)))

    fitted_signal           = util.pad_odd(fitted_signal, args.padding_factor)
    fitted_signal_spec      = rfft(fitted_signal, len(fitted_signal))
    fitted_signal_mag_spec  = np.abs(fitted_signal_spec)
    fitted_donor_signal     = util.pad_odd(fitted_donor_signal, args.padding_factor)
    fitted_donor_spec       = rfft(fitted_donor_signal, len(fitted_donor_signal))
    fft_length              = len(fitted_signal)
else:
    padded_singal = util.pad_odd(signal, args.padding_factor)
    spec          = rfft(padded_singal, len(padded_singal))
    mag_spec      = np.abs(spec)
    fft_length    = len(padded_singal)

Path(args.out_dir).mkdir(parents=True, exist_ok=True)
input_file_base = Path(args.input_file).stem

suffix = ""
needs_envelope = True
processed_spec = None
env_old = False
if args.mode == "real_spec":
    processed_spec = sign_real_spec(mag_spec, spec)
    needs_envelope = False
elif args.mode == "signal":
    processed_spec = sign_signal(mag_spec, signal)
elif args.mode == "chirp":
    if args.postscale:
        zero_phase_signal = irfft(mag_spec, fft_length)
        zero_phase_signal *= util.zero_phase_gain_comp(len(zero_phase_signal), 0.05)
        zero_phase_signal *= util.sine_envelope(len(zero_phase_signal), util.calc_transient_length(zero_phase_signal))
        zero_phase_spec = rfft(zero_phase_signal, fft_length)
        processed_spec = sign_chirp(zero_phase_spec, args.f0, args.f1, fs)
        suffix = f"{args.f0}-{args.f1}-post"
    else:
        processed_spec = sign_chirp(mag_spec, args.f0, args.f1, fs)
        suffix = f"{args.f0}-{args.f1}"
elif args.mode == "pwm":
    processed_spec = sign_pwm(mag_spec, args.f, args.duty0, args.duty1, fs)
    suffix = f"{args.f}-{args.duty0}-{args.duty1}"
elif args.mode == "donor_signal":
    if args.postscale:
        zero_phase_signal = irfft(fitted_signal_mag_spec, fft_length)
        zero_phase_signal *= util.zero_phase_gain_comp(len(zero_phase_signal), 0.05)
        zero_phase_signal *= util.sine_envelope(len(zero_phase_signal), util.calc_transient_length(zero_phase_signal))
        zero_phase_spec = rfft(zero_phase_signal, fft_length)
        processed_spec = sign_signal(zero_phase_spec, fitted_donor_signal)
        suffix = f"{Path(args.donor).stem}-post"
    else:
        processed_spec = sign_signal(fitted_signal_mag_spec, fitted_donor_signal)
        suffix = Path(args.donor).stem
elif args.mode == "donor_real_spec":
    processed_spec = sign_real_spec(fitted_signal_mag_spec, fitted_donor_spec)
    suffix = Path(args.donor).stem
    needs_envelope = False

processed_signal = irfft(processed_spec, fft_length)
if needs_envelope:
    processed_signal *= util.sine_envelope(len(processed_signal), util.calc_transient_length(processed_signal))
if suffix != "":
    suffix = f"_{suffix}"
util.write_wav(f"{args.out_dir}/{input_file_base}-{args.mode}{suffix}.wav", fs, processed_signal, normalize=True)
