import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})

n = 599
p = 16
m = (n-1) // 2
r = m//p
c = m % p

spec = np.ones(n)

for i in range(p//2, m, p):
    for j in range(0, p//2):
        spec[i+j] *= -1

for i in range(m):
    spec[m+1+i] = spec[m-i-1]

def dirichlet(k, x):
    return np.sin(k*np.pi*x/n) / np.sin(np.pi*x/n)

dp2  = np.zeros(n)
dr   = np.zeros(n)
dcp2 = np.zeros(n)

for i in range(n):
    dp2[i]  = dirichlet(p/2, i)
    dr[i]   = dirichlet(r, p*i)
    dcp2[i] = dirichlet(c-p/2, i)

s = np.zeros(n)
t = np.zeros(n)

for i in range(n):
    phi = i * np.pi / n
    s[i]  = dp2[i] * dr[i] * (np.cos((p*r - p + p/2 - 1)*phi) - np.cos((p*r + p/2 - 1)*phi))
    t[i]  = dp2[i] * np.cos((2*p*r+p/2-1)*phi) - dcp2[i] * np.cos((2*p*r + c + p/2 - 1)*phi)

h = 1/n * (spec[0] + 2*s*t)

plt.figure(figsize=(9, 10))
plt.subplot(4, 1, 1)
plt.title("Rectangular Frequency Spectrum")
plt.plot(spec, "bo", markersize=0.5)
plt.xlabel("Bin Index (k)")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 2)
plt.title("Dirichlet Kernels")
plt.plot(dr, "-bo", markersize=1.5, linewidth=0.5, label="$D_R(n)$")
plt.plot(dcp2, "ro", markersize=1.5, label="$D_{C-P/2}(n)$")
plt.plot(dp2, "go", markersize=1.5, label="$D_{P/2}$(n)")
plt.xlabel("Sample Index (n)")
plt.ylabel("Amplitude")
plt.legend(fontsize="small", loc="lower left")

plt.subplot(4, 1, 3)
plt.title("$S(n)$ and $T(n)$")
plt.plot(s, "-bo", markersize=1.5, linewidth=0.5)
plt.plot(t, "ro", markersize=1.5)
plt.xlabel("Sample Index (n)")
plt.ylabel("Amplitude")
plt.legend(["$S(n)$", "$T(n)$"], loc="lower center")

plt.subplot(4, 1, 4)
plt.title("$h_\\varsigma(n)$")
plt.plot(h, "-bo", markersize=1.5, linewidth=0.5)
plt.xlabel("Sample Index (n)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
