import numpy as np
import sounddevice as sd
from keras import models

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


def train_authentication_model(x_train, y_train, features, frames=100, epochs=15):
    x_train = x_train.reshape(x_train.shape[0], frames, features, 1)
    model = cnn_models.get_seventh_model(input_shape=(frames, features, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs)
    print(history.history)
    model.save('model')


if __name__ == '__main__':
    # Количество фреймов в одном шаблоне
    num_frames = 100
    # Количество MFCC
    num_mfcc = 13
    # Будут ли считаться первые и вторые производные
    use_deltas = True
    # Количество коэффициентов в одном фрейме
    num_features = (num_mfcc - 1) * 3 if use_deltas else num_mfcc - 1
    # Количество эпох обучения
    num_epochs = 15

    # При первом запуске программы
    # TODO: use_deltas in load_speakers_data
    # (x, y), (_, _) = load_speakers_data(num_registered=0, num_unregistered=10, num_train_files=2, num_test_files=0)
    # save_data_to_files(x, y)
    # train_model(x, y, num_epochs=3)

    print("Type 'r' if you want to register, 'l' if you want to log in: ")
    answer = input()
    if answer == 'r':
        sample_rate, signal = record(20)
        (x, y), (_, _) = load_data_from_files('data/x_train.npy', 'data/y_train.npy')
        templates, answers = load_templates_with_out(sample_rate, signal, 1, num_frames, num_mfcc, use_deltas)
        x = np.append(x, templates, axis=0)
        y = np.append(y, answers, axis=0)
        save_data_to_files(x, y)
        train_authentication_model(x, y, num_features, num_frames, num_epochs)
    elif answer == 'l':
        sample_rate, signal = record(5)
        # (sample_rate, signal) = wavfile.read("ru_0051.wav")
        templates = load_templates(sample_rate, signal, num_frames, num_mfcc, use_deltas)
        templates = templates.reshape(templates.shape[0], num_frames, num_features, 1)
        network_model = models.load_model('model')
        y_pred = network_model.predict(templates)
        success = True
        for i in range(len(templates)):
            print("Predicted=%s" % (y_pred[i]))
            if y_pred[i] <= 0.5:
                success = False
        if success:
            print("Success!")
        else:
            print("You are not registered")
