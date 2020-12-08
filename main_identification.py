from os import listdir
from os.path import join

import numpy
import sounddevice as sd
from keras import models
from scipy.io import wavfile
from tensorflow.python.keras.utils.np_utils import to_categorical

import cnn_models
from loaddata import save_data_to_files, load_templates, load_templates_with_out, load_data_from_files


def record(seconds):
    rate = 16000  # Sample rate

    print('Type enter when you will be ready to record an audio during %d seconds' % seconds)
    input()
    print('Recording...')
    recording = sd.rec(int(seconds * rate), samplerate=rate, channels=1)
    sd.wait()
    print('Done!')

    return rate, recording


def train_identification_model(x_train, y_train, num_classes, features, frames=100, epochs=15):
    x_train = x_train.reshape(x_train.shape[0], frames, features, 1)
    y_train_categorical = to_categorical(y_train)

    model = cnn_models.get_third_model(input_shape=(frames, features, 1), num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train_categorical, epochs=epochs)
    print(history.history)
    model.save('model')


if __name__ == '__main__':
    # Количество фреймов в одном шаблоне
    num_frames = 50
    # Количество MFCC
    num_mfcc = 25
    # Будут ли считаться первые и вторые производные
    use_deltas = True
    # Количество коэффициентов в одном фрейме
    num_features = (num_mfcc - 1) * 3 if use_deltas else num_mfcc - 1
    # Количество эпох обучения
    num_epochs = 40
    # Количество зарегистрированных дикторов
    num_speakers = 50
    num_female = 12
    num_male = num_speakers - num_female

    speakers = sorted([
        join("speakers", "russian", "male", d) for d in listdir(join("speakers", "russian", "male"))])[:num_male]
    speakers.extend(sorted([
        join("speakers", "russian", "female", d) for d in listdir(join("speakers", "russian", "female"))])[:num_female])

    # При первом запуске программы
    # (x, y), (_, _) = load_templates_from_directories(directories=speakers, num_frames=num_frames, num_mfcc=num_mfcc,
    #                                                  use_deltas=use_deltas, num_train_templates=20,
    #                                                  num_test_templates=4)

    # save_data_to_files(x, y)
    # train_identification_model(x, y, num_speakers, num_features, num_frames, num_epochs)

    print("Type 'r' if you want to register, 'l' if you want to log in: ")
    # TODO: type login and save it in speakers
    answer = input()
    if answer == 'r':
        sample_rate, signal = record(20)
        (x, y), (_, _) = load_data_from_files('data/x_train.npy', 'data/y_train.npy')
        templates, answers = load_templates_with_out(sample_rate, signal, num_speakers, num_frames, num_mfcc,
                                                     use_deltas)
        x = numpy.append(x, templates, axis=0)
        y = numpy.append(y, answers, axis=0)
        save_data_to_files(x, y)
        num_speakers += 1
        train_identification_model(x, y, num_speakers, num_features, num_frames, num_epochs)
    elif answer == 'l':
        # sample_rate, signal = record(5)
        # (sample_rate, signal) = wavfile.read("speakers/russian/male/vsh/vsh_ru_0009.wav")
        (sample_rate, signal) = wavfile.read("speakers/russian/male/NazarovMarat/ru_0039.wav")
        templates = load_templates(sample_rate, signal, num_frames, num_mfcc, use_deltas)
        templates = templates.reshape(templates.shape[0], num_frames, num_features, 1)
        network_model = models.load_model('model')
        predicted = numpy.argmax(network_model.predict(templates), axis=1)
        for p in predicted:
            print("Predicted id = %d, login = %s" % (p, speakers[p]))
