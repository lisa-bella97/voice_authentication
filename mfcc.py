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
