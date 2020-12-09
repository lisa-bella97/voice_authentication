import os
from collections import Counter
from os import listdir
from os.path import join

import numpy
from keras import models
from scipy.io import wavfile

from main import train_identification_model
from loaddata import load_templates, load_templates_from_directories, save_data_to_files

if __name__ == '__main__':
    # tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
    # Количество фреймов в одном шаблоне
    num_frames = 50
    # Количество MFCC
    num_mfcc = 25
    # Будут ли считаться первые и вторые производные
    use_deltas = True
    # Количество коэффициентов в одном фрейме
    num_features = (num_mfcc - 1) * 3 if use_deltas else num_mfcc - 1
    # Количество эпох обучения
    num_epochs = 20
    # Количество зарегистрированных дикторов
    num_speakers = 50
    num_female = 9
    num_male = num_speakers - num_female

    speakers = sorted([
        join("speakers", "russian", "male", d) for d in listdir(join("speakers", "russian", "male"))])[:num_male]
    speakers.extend(sorted([
        join("speakers", "russian", "female", d) for d in listdir(join("speakers", "russian", "female"))])[:num_female])

    # При первом запуске программы
    (x, y), (_, _) = load_templates_from_directories(directories=speakers, num_frames=num_frames, num_mfcc=num_mfcc,
                                                     use_deltas=use_deltas, num_train_templates=20,
                                                     num_test_templates=4)
    save_data_to_files(x, 'data/x_identification.npy', y, 'data/y_identification.npy')
    train_identification_model(x, y, num_speakers, num_features, num_frames, num_epochs)
    #

    network_model = models.load_model('model')

    for i, speaker_directory in enumerate(speakers):
        speaker_files = sorted([join(speaker_directory, f) for f in listdir(speaker_directory)])
        for file in speaker_files:
            (sample_rate, signal) = wavfile.read(file)
            templates = load_templates(sample_rate, signal, num_frames, num_mfcc, use_deltas)
            templates = templates.reshape(templates.shape[0], num_frames, num_features, 1)
            predicted = numpy.argmax(network_model.predict(templates), axis=1)
            main_class = Counter(predicted).most_common()[0]
            border = len(predicted) // 2 + 1
            if main_class[1] < border:
                print("Less than a half answers. Predicted = %s, actual file = %s" % (speakers[main_class[0]], file))
            if speakers[main_class[0]] != speaker_directory:
                print("Wrong answer. Predicted = %s, actual file = %s" % (speakers[main_class[0]], file))
                new_path = os.path.abspath(file).replace('speakers', 'bad_speakers')
                if not os.path.exists(os.path.dirname(new_path)):
                    os.makedirs(os.path.dirname(new_path))
                os.rename(file, new_path)
