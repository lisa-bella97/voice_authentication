import os

# To disable logs from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
from collections import Counter
from os import listdir
from os.path import join

import numpy
import sounddevice as sd
from keras import models
from tensorflow.python.keras.utils.np_utils import to_categorical

import cnn_models
from loaddata import save_data_to_files, load_templates, load_templates_with_out, load_data_from_files, \
    load_speakers_data_authentication, load_templates_from_directories


def record(seconds):
    """Records an audio during some seconds.

    Sample rate of recording is always 16000, one channel.

    Parameters
    ----------
    seconds : int
        Number of seconds in recording.

    Returns
    -------
    rate : int
        Sample rate of recording.
    recording : numpy array
        Recorded data.

    """
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
    model = cnn_models.get_second_model(input_shape=(frames, features, 1))
    print(model.fit(x_train, y_train, epochs=epochs).history)
    model.save('authentication_model')


def train_identification_model(x_train, y_train, num_classes, features, frames=100, epochs=15):
    x_train = x_train.reshape(x_train.shape[0], frames, features, 1)
    model = cnn_models.get_third_model(input_shape=(frames, features, 1), num_classes=num_classes)
    print(model.fit(x_train, to_categorical(y_train), epochs=epochs).history)
    model.save('identification_model')


def get_users():
    with open('data/users.txt', 'rb') as fp:
        return pickle.load(fp)


def save_users(users_list):
    if os.path.isfile('data/users.txt'):
        os.remove('data/users.txt')
    with open('data/users.txt', 'wb') as fp:
        pickle.dump(users_list, fp)


# Number of frames in one template
frames_authentication = 50
frames_identification = 25
# MFCC number
num_mfcc = 31
# Will the first and second derivatives be calculated
use_deltas = False
# Number of coefficients in one frame
num_features = (num_mfcc - 1) * 3 if use_deltas else num_mfcc - 1
# Number of learning epochs
epochs_authentication = 40
epochs_identification = 15
# Number of templates for one user for training
train_templates_authentication = 24
train_templates_identification = 48


def prepare():
    (x_authentication, y_authentication), (_, _) = \
        load_speakers_data_authentication(num_frames=frames_authentication, num_mfcc=num_mfcc,
                                          use_deltas=use_deltas, num_registered_male=40, num_registered_female=5,
                                          num_unregistered_male=41, num_unregistered_female=4,
                                          num_train_templates=train_templates_authentication, num_test_templates=0)
    save_data_to_files(x_authentication, 'data/x_authentication.npy', y_authentication, 'data/y_authentication.npy')
    train_authentication_model(x_authentication, y_authentication, num_features, frames_authentication,
                               epochs_authentication)

    # registered speakers
    num_speakers = 45
    num_female = 5
    num_male = num_speakers - num_female

    speakers = sorted([join('speakers', 'russian', 'male', d)
                       for d in listdir(join('speakers', 'russian', 'male'))])[:num_male]
    speakers.extend(sorted([join('speakers', 'russian', 'female', d)
                            for d in listdir(join('speakers', 'russian', 'female'))])[:num_female])

    (x_identification, y_identification), (_, _) = \
        load_templates_from_directories(directories=speakers, num_frames=frames_identification, num_mfcc=num_mfcc,
                                        use_deltas=use_deltas, num_train_templates=train_templates_identification,
                                        num_test_templates=0)

    save_data_to_files(x_identification, 'data/x_identification.npy', y_identification, 'data/y_identification.npy')
    save_users(speakers)
    train_identification_model(x_identification, y_identification, num_speakers, num_features,
                               frames_identification, epochs_identification)


def register_user(nickname):
    users = get_users()
    if nickname in users:
        return False

    user_index = len(users)
    users.append(nickname)
    save_users(users)

    sample_rate, signal = record(20)

    (x_authentication, y_authentication), (_, _) = \
        load_data_from_files('data/x_authentication.npy', 'data/y_authentication.npy')
    templates_authentication, answers_authentication = \
        load_templates_with_out(sample_rate, signal, 1, frames_authentication, num_mfcc, use_deltas)
    if len(templates_authentication) < train_templates_authentication:
        raise RuntimeError('Too few templates in signal. Please make less pauses in speech')

    x_authentication = numpy.append(x_authentication, templates_authentication[:train_templates_authentication],
                                    axis=0)
    y_authentication = numpy.append(y_authentication, answers_authentication[:train_templates_authentication],
                                    axis=0)
    save_data_to_files(x_authentication, 'data/x_authentication.npy', y_authentication, 'data/y_authentication.npy')
    train_authentication_model(x_authentication, y_authentication, num_features, frames_authentication,
                               epochs_authentication)

    (x_identification, y_identification), (_, _) = \
        load_data_from_files('data/x_identification.npy', 'data/y_identification.npy')
    templates_identification, answers_identification = \
        load_templates_with_out(sample_rate, signal, user_index, frames_identification, num_mfcc, use_deltas)
    x_identification = numpy.append(x_identification, templates_identification[:train_templates_identification],
                                    axis=0)
    y_identification = numpy.append(y_identification, answers_identification[:train_templates_identification],
                                    axis=0)
    save_data_to_files(x_identification, 'data/x_identification.npy', y_identification, 'data/y_identification.npy')
    train_identification_model(x_identification, y_identification, len(users), num_features, frames_identification,
                               epochs_identification)

    return True


def login_user():
    sample_rate, signal = record(5)
    templates = load_templates(sample_rate, signal, frames_authentication, num_mfcc, use_deltas)
    templates = templates.reshape(templates.shape[0], frames_authentication, num_features, 1)
    network_model = models.load_model('authentication_model')

    y_pred = network_model.predict(templates)
    for i in range(len(templates)):
        print('Predicted=%f' % (y_pred[i]))
    mean_y_pred = y_pred.mean().round(decimals=1)
    print('Mean=%f' % mean_y_pred)

    if mean_y_pred >= 0.5:
        print('Success!')
        users = get_users()
        templates = load_templates(sample_rate, signal, frames_identification, num_mfcc, use_deltas)
        templates = templates.reshape(templates.shape[0], frames_identification, num_features, 1)
        network_model = models.load_model('identification_model')

        predicted = numpy.argmax(network_model.predict(templates), axis=1)
        for p in predicted:
            print('Predicted id = %d, login = %s' % (p, users[p]))
        main_class = Counter(predicted).most_common()[0]
        border = len(predicted) // 2 + 1
        if main_class[1] < border:
            print('You cannot be identified. You are not registered')
            return False
        else:
            print('Hello, %s!' % users[main_class[0]])
            return True
    else:
        print('You are not registered')
        return False


if __name__ == '__main__':
    # When you first start the program, set this flag to True
    is_first_launch = True

    try:
        if is_first_launch:
            print('Preparing system...')
            prepare()

        print('Type \'r\' if you want to register, \'l\' if you want to log in:')
        answer = input()

        if answer == 'r':
            print('Create your login:')
            login = input()
            register_user(login)
        elif answer == 'l':
            login_user()
        else:
            print('Wrong command')
    except Exception as e:
        print(e)

    print('\nType enter or close the window to exit')
    input()
