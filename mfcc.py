import numpy as np
from python_speech_features import mfcc, delta
from scipy.io import wavfile
import matplotlib.pyplot as plt

from preprocessing import remove_pauses, normalize_signal
from utils import stem


def get_mfcc_features(sample_rate, signal, num_coefficients=13, use_deltas=True, show_log=False):
    n_fft = 512
    n_mels = 40
    mfcc_features = mfcc(signal, samplerate=sample_rate, numcep=num_coefficients, winlen=n_fft / sample_rate,
                         winstep=0.01, nfft=n_fft, nfilt=n_mels, preemph=0.0, ceplifter=0,
                         appendEnergy=False, winfunc=np.hamming)

    if use_deltas:
        mfcc_features_d = delta(mfcc_features, 2)
        mfcc_features_dd = delta(mfcc_features_d, 2)
        mfcc_all_features = np.column_stack((mfcc_features[:, 1:], mfcc_features_d[:, 1:], mfcc_features_dd[:, 1:]))
    else:
        mfcc_all_features = mfcc_features[:, 1:]

    if show_log:
        print("mfcc_all_features.shape: ", mfcc_all_features.shape)
        print("mfcc_all_features:\n", mfcc_all_features)

    return mfcc_all_features


def get_mfcc_features_with_mean(wav_filename, num_coefficients=13, show_log=True):
    (sample_rate, signal) = wavfile.read(wav_filename)
    n_fft = 512
    n_mels = 40
    mfcc_features = mfcc(signal, samplerate=sample_rate, numcep=num_coefficients, winlen=n_fft / sample_rate,
                         winstep=0.01, nfft=n_fft, nfilt=n_mels, preemph=0.0, ceplifter=0,
                         appendEnergy=False, winfunc=np.hamming)

    mfcc_features_transposed = np.transpose(mfcc_features)[1:]
    mfcc_features_transposed_mean = np.round(mfcc_features_transposed.mean(axis=1), 3)
    if show_log:
        print("mfcc_features_transposed.shape: ", mfcc_features_transposed.shape)
        print("mfcc_features_transposed:\n", mfcc_features_transposed)
        print("mfcc_features_transposed_mean.shape:\n", mfcc_features_transposed_mean.shape)
        print("mfcc_features_transposed_mean:\n", mfcc_features_transposed_mean)

    mfcc_deltas = delta(mfcc_features, 2)
    mfcc_deltas_transposed = np.transpose(mfcc_deltas)[1:]
    mfcc_deltas_transposed_mean = np.round(mfcc_deltas_transposed.mean(axis=1), 3)
    if show_log:
        print("mfcc_deltas_transposed.shape: ", mfcc_deltas_transposed.shape)
        print("mfcc_deltas_transposed:\n", mfcc_deltas_transposed)
        print("mfcc_deltas_transposed_mean.shape:\n", mfcc_deltas_transposed_mean.shape)
        print("mfcc_deltas_transposed_mean:\n", mfcc_deltas_transposed_mean)

    mfcc_deltas_deltas_transposed = np.transpose(delta(mfcc_deltas, 2))[1:]
    mfcc_deltas_deltas_transposed_mean = np.round(mfcc_deltas_deltas_transposed.mean(axis=1), 3)
    if show_log:
        print("mfcc_deltas_deltas_transposed.shape: ", mfcc_deltas_deltas_transposed.shape)
        print("mfcc_deltas_deltas_transposed:\n", mfcc_deltas_deltas_transposed)
        print("mfcc_deltas_deltas_transposed_mean.shape:\n", mfcc_deltas_deltas_transposed_mean.shape)
        print("mfcc_deltas_deltas_transposed_mean:\n", mfcc_deltas_deltas_transposed_mean)

    return (mfcc_features_transposed, mfcc_features_transposed_mean), \
           (mfcc_deltas_transposed, mfcc_deltas_transposed_mean), \
           (mfcc_deltas_deltas_transposed, mfcc_deltas_deltas_transposed_mean)


if __name__ == '__main__':
    num_mfcc = 13
    use_deltas = True
    (sample_rate, signal) = wavfile.read("speakers/russian/female/anonymous104/ru_0036.wav")
    samples_without_pauses = remove_pauses(sample_rate, normalize_signal(signal))
    mfcc_features1 = get_mfcc_features(sample_rate, samples_without_pauses, num_mfcc, use_deltas)
    (sample_rate, signal) = wavfile.read("speakers/russian/female/anonymous104/ru_0037.wav")
    samples_without_pauses = remove_pauses(sample_rate, normalize_signal(signal))
    mfcc_features2 = get_mfcc_features(sample_rate, samples_without_pauses, num_mfcc, use_deltas)
    # Проверка значений коэффициентов в каких-либо фреймах
    plt.subplot(2, 1, 1)
    stem(mfcc_features1[0, :num_mfcc - 1], linefmt='r', markerfmt='ro')
    stem(mfcc_features1[1, :num_mfcc - 1], linefmt='b', markerfmt='bo')
    stem(mfcc_features1[20, :num_mfcc - 1], linefmt='y', markerfmt='yo')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    stem(mfcc_features2[0, :num_mfcc - 1], linefmt='r', markerfmt='ro')
    stem(mfcc_features2[1, :num_mfcc - 1], linefmt='b', markerfmt='bo')
    stem(mfcc_features2[20, :num_mfcc - 1], linefmt='y', markerfmt='yo')
    plt.grid(True)
    plt.show()
