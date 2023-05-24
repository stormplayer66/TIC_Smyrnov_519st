import numpy
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy import fft
from scipy import signal

N = 500
Frq = 1000
Frq_M = 29
Frq_F = 36
Random = numpy.random.normal(0, 10, N)
T_L_ox = numpy.arange(N) / Frq
W = Frq_M / (Frq / 2)
para_filter = scipy.signal.butter(3, W, 'low', output='sos')
filt_signal = scipy.signal.sosfiltfilt(para_filter, Random)

fi, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot(T_L_ox, filt_signal, linewidth = 1)
ax.set_xlabel("Час (секунди)", fontsize = 14)
ax.set_ylabel("Амплітуда сигналу", fontsize = 14)
plt.title("Сигнал із максимальною частотою Frq_M=29", fontsize = 14)
ax.grid()
fi.savefig('./figures/' + 'Графік 1' + '.png', dpi = 600)
dpi = 600

Spectrum = scipy.fft.fft(filt_signal)
Spectrum = numpy.abs(scipy.fft.fftshift(Spectrum))
length_signal = N
frequency_countdown = scipy.fft.fftfreq(length_signal, 1 / length_signal)
frequency_countdown = scipy.fft.fftshift(frequency_countdown)

fi, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot(frequency_countdown, Spectrum, linewidth = 1)
ax.set_xlabel("Частота (Гц)", fontsize = 14)
ax.set_ylabel("Амплітуда спектру", fontsize = 14)
plt.title("Спектр сигналу з максимальною частотою Frq_M=29", fontsize = 14)
ax.grid()
fi.savefig('./figures/' + 'графік 2' + '.png', dpi=600)

discrete_signals = []
steps = (2, 4, 8, 16)
for Dt in steps:
    discrete_signal = numpy.zeros(N)
    for i in range(0, round(N / Dt)):
        discrete_signal[i * Dt] = filt_signal[i * Dt]
    discrete_signals.append(list(discrete_signal))

fi, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(T_L_ox, discrete_signals[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fi.supxlabel('секунди', fontsize=14)
fi.supylabel('Амплітуда сигналу', fontsize=14)
fi.suptitle('Сигнал із кроком дискретизації', fontsize=14)
fi.savefig('./figures/' + 'графік 3' + '.png', dpi=600)

discrete_spectrums = []
for Ds in discrete_signals:
    Spectrum = fft.fft(Ds)
    Spectrum = numpy.abs(fft.fftshift(Spectrum))
    discrete_spectrums.append(list(Spectrum))

frequency_countdown = fft.fftfreq(N, 1 / N)
frequency_countdown = fft.fftshift(frequency_countdown)
fi, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(frequency_countdown, discrete_spectrums[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fi.supxlabel('Частота (Гц) ', fontsize=14)
fi.supylabel('Амплітуда спектра', fontsize=14)
fi.suptitle('Спектри сигналів із кроком дискретизації', fontsize=14)
fi.savefig('./figures/' + 'графік 4' + '.png', dpi=600)
W = Frq_F / (Frq / 2)
para_filter = signal.butter(3, W, 'low', output='sos')
filtered_discretes_signal = []
for discrete_signal in discrete_signals:
    discrete_signal = signal.sosfiltfilt(para_filter, discrete_signal)
    filtered_discretes_signal.append(list(discrete_signal))
fi, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(T_L_ox, filtered_discretes_signal[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fi.supxlabel('секунди', fontsize=14)
fi.supylabel('Амплітуда сигнала', fontsize=14)
fi.suptitle('Відновлені аналогові сигнали з кроком дискретизації', fontsize=14)
fi.savefig('./figures/' + 'графік 5' + '.png', dpi=600)
dispersions = []
signal_noise = []
for i in range(len(steps)):
    E1 = filtered_discretes_signal[i] - filt_signal
    dispersion = numpy.var(E1)
    dispersions.append(dispersion)
    signal_noise.append(numpy.var(filt_signal) / dispersion)
fi, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot(steps, dispersions, linewidth=1)
ax.set_xlabel('Крок дискретизації', fontsize=14)
ax.set_ylabel('Дісперсія', fontsize=14)
plt.title('Залежність дисперсії від кроку дискретизації', fontsize=14)
ax.grid()
fi.savefig('./figures/' + 'график 6' + '.png', dpi=600)
fi, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot(steps, signal_noise, linewidth=1)
ax.set_xlabel('Крок дискретизації', fontsize=14)
ax.set_ylabel('ССШ', fontsize=14)
plt.title('Залежність співвідношення сигнал-шум від кроку дискретизації', fontsize=14)
ax.grid()
fi.savefig('./figures/' + 'графік 7' + '.png', dpi=600)
# практичне заняття 4
bits_list = []
quantize_signals = []
num = 0
for M in [4, 16, 64, 256]:
    delta = (numpy.max(filt_signal) - numpy.min(filt_signal)) / (M - 1)
    quantize_signal = delta * np.round(filt_signal / delta)
    quantize_signals.append(list(quantize_signal))
    quantize_levels = numpy.arange(numpy.min(quantize_signal), numpy.max(quantize_signal) + 1, delta)
    quantize_bit = numpy.arange(0, M)
    quantize_bit = [format(bits, '0' + str(int(numpy.log(M) / numpy.log(2))) + 'b') for bits in quantize_bit]
    quantize_table = numpy.c_[quantize_levels[:M], quantize_bit[:M]]
    fi, ax = plt.subplots(figsize=(14 / 2.54, M / 2.54))
    table = ax.table(cellText=quantize_table, colLabels=['Значення сигналу', 'Кодова послідовність'], loc='center')
    table.set_fontsize(14)
    table.scale(1, 2)
    ax.axis('off')
    fi.savefig('./figures/' + 'Таблиця квантування для %d рівнів' % M + '.png', dpi=600)
    bits = []
    for signal_value in quantize_signal:
        for index, value in enumerate(quantize_levels[:M]):
            if numpy.round(numpy.abs(signal_value - value), 0) == 0:
                bits.append(quantize_bit[index])
                break

    bits = [int(item) for item in list(''.join(bits))]
    bits_list.append(bits)
    fi, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.step(numpy.arange(0, len(bits)), bits, linewidth=0.1)
    ax.set_xlabel('Бити', fontsize=14)
    ax.set_ylabel('Амплітуда сигналу', fontsize=14)
    plt.title('Кодова послідовність при кількості рівнів квантування {M}', fontsize=14)
    ax.grid()
    fi.savefig('./figures/' + 'Графік %d ' % (8 + num) + '.png', dpi=600)
    num += 1
dispersions = []
signal_noise = []
for i in range(4):
    E1 = quantize_signals[i] - filt_signal
    dispersion = numpy.var(E1)
    dispersions.append(dispersion)
    signal_noise.append(numpy.var(filt_signal) / dispersion)
fi, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(T_L_ox, quantize_signals[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fi.supxlabel('Час (секунди)', fontsize=14)
fi.supylabel('Амплітуда сигналу', fontsize=14)
fi.suptitle(f'Цифрові сигнали з рівнями квантування (4, 16, 64, 256)', fontsize=14)
fi.savefig('./figures/' + 'графік 12' + '.png', dpi=600)
fi, ax = plt.subplots (figsize = (21 / 2.54, 14 / 2.54))
ax.plot([4, 16, 64, 256], dispersions, linewidth=1)
ax.set_xlabel('Кількість рівнів квантування', fontsize=14)
ax.set_ylabel('Дисперсія', fontsize=14)
plt.title(f'Залежність дисперсії від кількості рівнів квантування', fontsize=14)
ax.grid()
fi.savefig('./figures/' + 'графік 13' + '.png', dpi=600)
fi, ax = plt.subplots (figsize = (21 / 2.54, 14 / 2.54))
ax.plot([4, 16, 64, 256], signal_noise, linewidth=1)
ax.set_xlabel('Кількість рівнів квантування', fontsize=14)
ax.set_ylabel('ССШ', fontsize=14)
plt.title(f'Залежність співвідношення сигнал-шум від кількості рівнів квантування', fontsize=14)
ax.grid()
fi.savefig('./figures/' + 'графік 14' + '.png', dpi=600)