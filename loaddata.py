from os import listdir
from os.path import join

import numpy as np
from scipy.io import wavfile

from mfcc import get_mfcc_features
from preprocessing import remove_pauses, normalize_signal


# Загрузить шаблоны из файлов и присвоить им значение эталонного выхода out (0 или 1)
def load_templates_with_out(files, out, num_frames=100, num_mfcc=13):
    x = []
    y = []

    for file in files:
        (sample_rate, signal) = wavfile.read(file)
        samples_without_pauses = remove_pauses(sample_rate, normalize_signal(signal))
        mfcc_features = get_mfcc_features(sample_rate, samples_without_pauses, num_mfcc)
        templates = np.split(mfcc_features[:mfcc_features.shape[0] // num_frames * num_frames],
                             mfcc_features.shape[0] // num_frames)
        for template in templates:
            x.append(template)
            y.append(out)

    return x, y


def load_data(num_frames=100, num_mfcc=13, num_registered=5, num_unregistered=10, num_train_files=6, num_test_files=2):
    speakers = sorted([join("speakers", d) for d in listdir("speakers")])
    train_files_registered, test_files_registered, train_files_unregistered, test_files_unregistered = [], [], [], []

    for speaker in speakers[:num_registered]:
        files_registered = sorted([join(speaker, "wav", f) for f in listdir(join(speaker, "wav"))])
        train_files_registered.extend(files_registered[:num_train_files])
        test_files_registered.extend(files_registered[num_train_files:num_train_files + num_test_files])

    for speaker in speakers[num_registered:num_registered + num_unregistered]:
        files_unregistered = sorted([join(speaker, "wav", f) for f in listdir(join(speaker, "wav"))])
        train_files_unregistered.extend(files_unregistered[:num_train_files])
        test_files_unregistered.extend(files_unregistered[num_train_files:num_train_files + num_test_files])

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    x, y = load_templates_with_out(train_files_registered, 1, num_frames, num_mfcc)
    x_train.extend(x)
    y_train.extend(y)

    x, y = load_templates_with_out(train_files_unregistered, 0, num_frames, num_mfcc)
    x_train.extend(x)
    y_train.extend(y)

    x, y = load_templates_with_out(test_files_registered, 1, num_frames, num_mfcc)
    x_test.extend(x)
    y_test.extend(y)

    x, y = load_templates_with_out(test_files_unregistered, 0, num_frames, num_mfcc)
    x_test.extend(x)
    y_test.extend(y)

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


def save_data_to_files(x_train, y_train, x_test, y_test):
    np.save('x_train', x_train)
    np.save('y_train', y_train)
    np.save('x_test', x_test)
    np.save('y_test', y_test)


def load_data_from_files(x_train, y_train, x_test, y_test):
    return (np.load(x_train), np.load(y_train)), (np.load(x_test), np.load(y_test))
