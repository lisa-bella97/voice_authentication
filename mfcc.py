import numpy as np
from python_speech_features import mfcc, delta
from scipy.io import wavfile


def get_mfcc_features(sample_rate, signal, num_coefficients=13, show_log=False):
    mfcc_features = mfcc(signal, samplerate=sample_rate, numcep=num_coefficients, winlen=0.025, winstep=0.01, nfft=512,
                         preemph=0, ceplifter=0, appendEnergy=False, winfunc=np.hamming)

    mfcc_features_d = delta(mfcc_features, 2)
    mfcc_features_dd = delta(mfcc_features_d, 2)
    mfcc_all_features = np.column_stack((mfcc_features[:, 1:], mfcc_features_d[:, 1:], mfcc_features_dd[:, 1:]))
    if show_log:
        print("mfcc_all_features.shape: ", mfcc_all_features.shape)
        print("mfcc_all_features:\n", mfcc_all_features)

    return mfcc_all_features


def get_mfcc_features_with_mean(wav_filename, num_coefficients=13, show_log=True):
    (sample_rate, signal) = wavfile.read(wav_filename)
    mfcc_features = mfcc(signal, samplerate=sample_rate, numcep=num_coefficients, preemph=0, ceplifter=0,
                         appendEnergy=False, winfunc=np.hamming)
    # mfcc_features = mfcc(signal, sample_rate, winlen=0.24, winstep=0.015, appendEnergy=False, preemph=0, ceplifter=0,
    # winfunc=np.hamming)

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

# Проверка средних значений коэффициентов
# def check_mean_values():
#     (liza1_mfcc, liza1_mfcc_mean), (liza1_deltas, liza1_deltas_mean), (
#         liza1_deltas_deltas, liza1_deltas_deltas_mean) = get_mfcc_features_with_mean("samples/pausesDeleted_Liza_1.wav")
#     (liza2_mfcc, liza2_mfcc_mean), (liza2_deltas, liza2_deltas_mean), (
#         liza2_deltas_deltas, liza2_deltas_deltas_mean) = get_mfcc_features_with_mean("samples/pausesDeleted_Liza_2.wav")
#     (test_mfcc, test_mfcc_mean), (test_deltas, test_deltas_mean), (
#         test_deltas_deltas, test_deltas_deltas_mean) = get_mfcc_features_with_mean(
#         "samples/ru_0075.wav")
#
#     plt.subplot(3, 1, 1)
#     stem(liza1_mfcc_mean, linefmt='r', markerfmt='ro')
#     stem(liza2_mfcc_mean, linefmt='g', markerfmt='go')
#     stem(test_mfcc_mean, linefmt='b', markerfmt='bo')
#     plt.grid(True)
#     plt.subplot(3, 1, 2)
#     stem(liza1_deltas_mean, linefmt='r', markerfmt='ro')
#     stem(liza2_deltas_mean, linefmt='g', markerfmt='go')
#     stem(test_deltas_mean, linefmt='b', markerfmt='bo')
#     plt.grid(True)
#     plt.subplot(3, 1, 3)
#     stem(liza1_deltas_deltas_mean, linefmt='r', markerfmt='ro')
#     stem(liza2_deltas_deltas_mean, linefmt='g', markerfmt='go')
#     stem(test_deltas_deltas_mean, linefmt='b', markerfmt='bo')
#     plt.grid(True)
#     plt.show()


# Проверка значений коэффициентов в определенных фреймах
# def check_specific_values():
#     num_coeff = 13
#     liza1 = get_mfcc_features("samples/pausesDeleted_Liza_1.wav", num_coefficients=num_coeff)
#     liza2 = get_mfcc_features("samples/pausesDeleted_Liza_2.wav")
#     test = get_mfcc_features("samples/pausesDeleted_Liza_2.wav")
#
#     # Проверка значений коэффициентов в каких-либо фреймах
#     plt.subplot(3, 1, 1)
#     stem(liza1[0, :num_coeff - 1], linefmt='r', markerfmt='ro')
#     stem(liza1[1, :num_coeff - 1], linefmt='b', markerfmt='bo')
#     stem(liza1[2, :num_coeff - 1], linefmt='y', markerfmt='yo')
#     plt.grid(True)
#     plt.subplot(3, 1, 2)
#     stem(liza2[0, :num_coeff - 1], linefmt='r', markerfmt='ro')
#     stem(liza2[1, :num_coeff - 1], linefmt='b', markerfmt='bo')
#     stem(liza2[2, :num_coeff - 1], linefmt='y', markerfmt='yo')
#     plt.grid(True)
#     plt.subplot(3, 1, 3)
#     stem(test[0, :num_coeff - 1], linefmt='r', markerfmt='ro')
#     stem(test[1, :num_coeff - 1], linefmt='b', markerfmt='bo')
#     stem(test[2, :num_coeff - 1], linefmt='y', markerfmt='yo')
#     plt.grid(True)
#     plt.show()
