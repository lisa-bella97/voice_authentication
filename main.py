import sounddevice as sd
import numpy as np
from keras import models

from loaddata import load_speakers_data, save_data_to_files, load_templates, load_templates_with_out, \
    load_data_from_files
from testcnn import get_seventh_model


def record(seconds):
    rate = 16000  # Sample rate

    print("Type enter when you will be ready to record an audio during " + str(seconds) + " seconds: ")
    input()
    print("Recording...")
    recording = sd.rec(int(seconds * rate), samplerate=rate, channels=1)
    sd.wait()
    print("Done!")

    return rate, recording


def train_model(x_train, y_train, frames=100, mfcc=13, num_epochs=15):
    num_features = (mfcc - 1) * 3
    x_train = x_train.reshape(x_train.shape[0], frames, num_features, 1)
    model = get_seventh_model(input_shape=(frames, num_features, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=num_epochs)
    print(history.history)
    model.save('model')


if __name__ == '__main__':
    # При первом запуске программы
    # (x, y), (_, _) = load_speakers_data(num_registered=0, num_unregistered=10, num_train_files=2, num_test_files=0)
    # save_data_to_files(x, y)
    # train_model(x, y, num_epochs=3)

    num_frames = 100
    num_mfcc = 13

    print("Type 'r' if you want to register, 'l' if you want to log in: ")
    answer = input()
    if answer == 'r':
        sample_rate, signal = record(20)
        (x, y), (_, _) = load_data_from_files('data/x_train.npy', 'data/y_train.npy')
        templates, answers = load_templates_with_out(sample_rate, signal, 1)
        x = np.append(x, templates, axis=0)
        y = np.append(y, answers, axis=0)
        save_data_to_files(x, y)
        train_model(x, y)
    elif answer == 'l':
        sample_rate, signal = record(5)
        templates = load_templates(sample_rate, signal)
        templates = templates.reshape(templates.shape[0], num_frames, (num_mfcc - 1) * 3, 1)
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
