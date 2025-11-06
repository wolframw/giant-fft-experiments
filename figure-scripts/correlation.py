import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, correlate
from scipy.fft import rfft

# ---------- config ----------
TITLE        = "Signal1 (Input) + Signal2 (Sign Donor)"
TITLE1       = f"{TITLE}: Signal Overlay"
TITLE2       = f"{TITLE}: Correlation"
LEGEND1      = "Donor"
LEGEND2      = "Resynth."
DONOR_FILE   = ""
RESYN_FILE   = ""
NOFIG        = False

# ---------- helpers ----------
def load_mono(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    m = np.max(np.abs(x))
    if m > 0: x = x / m
    return sr, x

def segment_corr(x, y, fs, segment_ms=20):
    seg_len = int(fs * segment_ms / 1000)
    corrs = np.zeros(len(x))
    x1 = np.pad(x, (0, len(x) + seg_len))
    y1 = np.pad(y, (0, len(y) + seg_len))

    for i in range(0, len(corrs)):
        seg_x = x1[max(0, i - seg_len//2) : min(len(x1), i+seg_len//2)]
        seg_y = y1[max(0, i - seg_len//2) : min(len(y1), i+seg_len//2)]
        # corrs[i] = correlate(seg_x, seg_y, mode='valid')
        corrs[i] = np.corrcoef(seg_x, seg_y)[0, 1]
    return corrs


# --------------------
sr_a, donor   = load_mono(DONOR_FILE)
sr_b, resynth = load_mono(RESYN_FILE)
assert sr_a == sr_b, "Sample rates must match"
sr = sr_a

resynth = resynth[:len(resynth)//2]

if len(donor) < len(resynth):
    resynth = resynth[:len(donor)]
elif len(donor) > len(resynth):
    donor = donor[:len(resynth)]

print(np.corrcoef(resynth, donor)[0, 1])

if not NOFIG:
    t_sig = np.arange(len(resynth)) / sr

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
    })

    plt.figure(figsize=(9, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t_sig, donor, label=LEGEND1, linewidth=0.8)
    plt.plot(t_sig, resynth, label=LEGEND2, alpha=0.7, linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.05, 1.05)
    # plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(TITLE1)
    plt.margins(x=0)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_sig, segment_corr(resynth, donor, sr_a, segment_ms=20), linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("Time (s)")
    plt.ylabel("Correlation")
    plt.title(TITLE2)
    plt.margins(x=0)
    plt.legend()

    plt.tight_layout()
    plt.show()
