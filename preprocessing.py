from math import log

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.io import wavfile
from sklearn.cluster import KMeans


def get_frame_powers(sample_rate, samples):
    # Определение окна для Фурье-преобразования
    N = samples.shape[0]  # Число отсчетов сигнала
    T = 1.0 / sample_rate  # Период дискретизации
    n = 0.02 / T  # Количество отсчетов в периоде стационарности
    s = log(n) // log(2)  # Степень двойки(log2(n) = log(n) / log(2))
    n_in_frame = int(2 ** s)  # Количество отсчетов в кадре
    K = int(N / n_in_frame)  # Количество итераций Фурье-преобразования

    frame_powers = np.array([[0.0, 0]] * K)

    for i in range(K):
        fft_result = fft(samples[i * n_in_frame:(i + 1) * n_in_frame])
        frame_powers[i] = [0.5 * (abs(fft_result) ** 2).sum(), 0]

    return frame_powers, n_in_frame


def remove_pauses(sample_rate, samples):
    frame_powers, n_in_frame = get_frame_powers(sample_rate, samples)

    init_centers = np.array([[frame_powers[:, 0].min(), 0],
                             [frame_powers[:, 0].mean(), 0],
                             [frame_powers[:, 0].max(), 0]])
    kmeans = KMeans(n_clusters=3, init=init_centers, n_init=1).fit(frame_powers)
    kmeans_labels = kmeans.labels_

    # Значения frame_powers, соответствующие нулевому кластеру (label = 0), или участкам паузы
    i, j = 0, 0
    pauses_power_cluster = np.zeros(kmeans_labels[kmeans_labels == 0].shape[0])
    for label in kmeans_labels:
        if label == 0:
            pauses_power_cluster[j] = frame_powers[i][0]
            j += 1
        i += 1

    # Критерий Неймана-Пирсона для улучшения определения участков пауз
    b = 0.01
    U = pauses_power_cluster.mean() / 10 + b * pauses_power_cluster.std()

    samples_without_pauses = []
    i = 0
    for label in kmeans_labels:
        if (label == 0 and frame_powers[i][0] > U) or label != 0:
            samples_without_pauses.extend(samples[i * n_in_frame:(i + 1) * n_in_frame])
        i += 1

    return np.array(samples_without_pauses)


def plot_signal(sample_rate, signal):
    plt.plot(np.linspace(0, len(signal) / sample_rate, num=len(signal)), signal)
    plt.grid(True)


def show_signal_and_power(sample_rate, signal, frame_powers):
    plt.subplot(2, 1, 1)
    plot_signal(sample_rate, signal)
    plt.subplot(2, 1, 2)
    plt.plot(frame_powers[:, 0])
    plt.grid(True)
    plt.show()


def pot_normalized_signal(sample_rate, signal):
    plot_signal(sample_rate, normalize_signal(signal))


def normalize_signal(signal):
    return signal / np.max(np.abs(signal))


if __name__ == '__main__':
    (rate, sig) = wavfile.read("speakers/13/wav/ru_0052.wav")
    normalized_sig = normalize_signal(sig)
    # show_signal_and_power(rate, normalized_sig, get_frame_powers(rate, normalized_sig)[0])
    signal_without_pauses = remove_pauses(rate, normalized_sig)
    # show_signal_and_power(rate, signal_without_pauses, get_frame_powers(rate, signal_without_pauses)[0])
    wavfile.write("speakers/13/wav/ru_0052_without_pauses.wav", rate, signal_without_pauses)
