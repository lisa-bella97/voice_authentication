import math
import os
from os import listdir
from os.path import join

import numpy as np
from scipy.io import wavfile

from mfcc import get_mfcc_features
from preprocessing import remove_pauses, normalize_signal


def load_templates(sample_rate, signal, num_frames=25, num_mfcc=13, use_deltas=True):
    x = []

    samples_without_pauses = remove_pauses(sample_rate, normalize_signal(signal))
    mfcc_features = get_mfcc_features(sample_rate, samples_without_pauses, num_mfcc, use_deltas)
    templates = np.split(mfcc_features[:mfcc_features.shape[0] // num_frames * num_frames],
                         mfcc_features.shape[0] // num_frames)

    for template in templates:
        x.append(template)

    return np.array(x)


def load_templates_with_out(sample_rate, signal, out, num_frames=25, num_mfcc=13, use_deltas=True):
    x = []
    y = []

    samples_without_pauses = remove_pauses(sample_rate, normalize_signal(signal))
    mfcc_features = get_mfcc_features(sample_rate, samples_without_pauses, num_mfcc, use_deltas)

    templates = np.split(mfcc_features[:mfcc_features.shape[0] // num_frames * num_frames],
                         mfcc_features.shape[0] // num_frames)

    for template in templates:
        x.append(template)
        y.append(out)

    return x, y


# Загрузить шаблоны из файлов и присвоить им значение эталонного выхода out
def load_templates_from_files_with_out(files, out, num_templates=math.inf, num_frames=25, num_mfcc=13,
                                       use_deltas=True):
    x = []
    y = []

    for file in files:
        (sample_rate, signal) = wavfile.read(file)
        x_file, y_file = load_templates_with_out(sample_rate, signal, out, num_frames, num_mfcc, use_deltas)
        x.extend(x_file)
        y.extend(y_file)
        if len(y) > num_templates:
            break

    if num_templates == math.inf:
        return x, y
    else:
        if len(y) < num_templates:
            raise RuntimeError(
                'num_templates ({}) is too big: use value less than {}\nDirectory: {}'.format(num_templates, len(y),
                                                                                              files))
        return x[:num_templates], y[:num_templates]


def load_speakers_data_authentication(num_frames=25, num_mfcc=13, use_deltas=True, num_registered_male=5,
                                      num_registered_female=5, num_unregistered_male=5, num_unregistered_female=5,
                                      num_train_templates=6, num_test_templates=4):
    male_speakers = sorted(
        [join("speakers", "russian", "male", d) for d in listdir(join("speakers", "russian", "male"))])
    female_speakers = sorted(
        [join("speakers", "russian", "female", d) for d in listdir(join("speakers", "russian", "female"))])
    x_train, y_train, x_test, y_test = [], [], [], []

    for speaker in male_speakers[:num_registered_male]:
        files_registered = sorted([join(speaker, f) for f in listdir(speaker)])
        x, y = load_templates_from_files_with_out(files=files_registered, out=1,
                                                  num_templates=num_train_templates + num_test_templates,
                                                  num_frames=num_frames, num_mfcc=num_mfcc, use_deltas=use_deltas)
        x_train.extend(x[:num_train_templates])
        y_train.extend(y[:num_train_templates])
        x_test.extend(x[num_train_templates:num_train_templates + num_test_templates])
        y_test.extend(y[num_train_templates:num_train_templates + num_test_templates])

    for speaker in male_speakers[num_registered_male:num_registered_male + num_unregistered_male]:
        files_unregistered = sorted([join(speaker, f) for f in listdir(speaker)])
        x, y = load_templates_from_files_with_out(files=files_unregistered, out=0,
                                                  num_templates=num_train_templates + num_test_templates,
                                                  num_frames=num_frames, num_mfcc=num_mfcc, use_deltas=use_deltas)
        x_train.extend(x[:num_train_templates])
        y_train.extend(y[:num_train_templates])
        x_test.extend(x[num_train_templates:num_train_templates + num_test_templates])
        y_test.extend(y[num_train_templates:num_train_templates + num_test_templates])

    for speaker in female_speakers[:num_registered_female]:
        files_registered = sorted([join(speaker, f) for f in listdir(speaker)])
        x, y = load_templates_from_files_with_out(files=files_registered, out=1,
                                                  num_templates=num_train_templates + num_test_templates,
                                                  num_frames=num_frames, num_mfcc=num_mfcc, use_deltas=use_deltas)
        x_train.extend(x[:num_train_templates])
        y_train.extend(y[:num_train_templates])
        x_test.extend(x[num_train_templates:num_train_templates + num_test_templates])
        y_test.extend(y[num_train_templates:num_train_templates + num_test_templates])

    for speaker in female_speakers[num_registered_female:num_registered_female + num_unregistered_female]:
        files_unregistered = sorted([join(speaker, f) for f in listdir(speaker)])
        x, y = load_templates_from_files_with_out(files=files_unregistered, out=0,
                                                  num_templates=num_train_templates + num_test_templates,
                                                  num_frames=num_frames, num_mfcc=num_mfcc, use_deltas=use_deltas)
        x_train.extend(x[:num_train_templates])
        y_train.extend(y[:num_train_templates])
        x_test.extend(x[num_train_templates:num_train_templates + num_test_templates])
        y_test.extend(y[num_train_templates:num_train_templates + num_test_templates])

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


def load_speakers_data_identification(num_frames=25, num_mfcc=13, use_deltas=True, num_speakers=50, num_female=9,
                                      num_train_templates=6, num_test_templates=4):
    num_male = num_speakers - num_female

    speakers = sorted([join("speakers", "russian", "male", d) for d in listdir(join("speakers", "russian", "male"))])[
               :num_male]
    speakers.extend(
        sorted([join("speakers", "russian", "female", d) for d in listdir(join("speakers", "russian", "female"))])[
        :num_female])

    return load_templates_from_directories(directories=speakers, num_frames=num_frames, num_mfcc=num_mfcc,
                                           use_deltas=use_deltas, num_train_templates=num_train_templates,
                                           num_test_templates=num_test_templates)


def load_templates_from_directories(directories, num_frames=25, num_mfcc=13, use_deltas=True, num_train_templates=10,
                                    num_test_templates=2):
    x_train, y_train, x_test, y_test = [], [], [], []

    for i, speaker_directory in enumerate(directories):
        speaker_files = sorted([join(speaker_directory, f) for f in listdir(speaker_directory)])
        half_len = len(speaker_files) // 2

        x, y = load_templates_from_files_with_out(files=speaker_files[:half_len], out=i,
                                                  num_templates=num_train_templates, num_frames=num_frames,
                                                  num_mfcc=num_mfcc, use_deltas=use_deltas)
        x_train.extend(x[:num_train_templates])
        y_train.extend(y[:num_train_templates])

        x, y = load_templates_from_files_with_out(files=speaker_files[half_len:], out=i,
                                                  num_templates=num_test_templates, num_frames=num_frames,
                                                  num_mfcc=num_mfcc, use_deltas=use_deltas)
        x_test.extend(x[:num_test_templates])
        y_test.extend(y[:num_test_templates])

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


def save_data_to_files(x_train, x_train_path, y_train, y_train_path, x_test=None, x_test_path=None, y_test=None,
                       y_test_path=None):
    os.remove(x_train_path)
    open(x_train_path, 'a').close()
    np.save(x_train_path, x_train)
    os.remove(y_train_path)
    open(y_train_path, 'a').close()
    np.save(y_train_path, y_train)
    if x_test is not None:
        os.remove(x_test_path)
        open(x_test_path, 'a').close()
        np.save(x_test_path, x_test)
    if y_test is not None:
        os.remove(y_test_path)
        open(y_test_path, 'a').close()
        np.save(y_test_path, y_test)


def load_data_from_files(x_train_path, y_train_path, x_test_path=None, y_test_path=None):
    return (np.load(x_train_path), np.load(y_train_path)), \
           (np.load(x_test_path) if x_test_path is not None else None,
            np.load(y_test_path) if y_test_path is not None else None)
