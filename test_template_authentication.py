import numpy
from prettytable import PrettyTable

import cnn_models
from loaddata import load_speakers_data_authentication, save_data_to_files


def get_model(mfcc=13, deltas=True, frames=25, train_templates=48):
    # Количество шаблонов для тестирования (на одного диктора) (60%:40%)
    num_test_templates = int(train_templates // 1.5)
    # Количество зарегистрированных дикторов
    num_registered = 50
    num_registered_female = 5
    num_registered_male = num_registered - num_registered_female
    # Количество незарегистрированных дикторов
    num_unregistered = 50
    num_unregistered_female = 4
    num_unregistered_male = num_unregistered - num_unregistered_female
    # Количество коэффициентов в одном фрейме
    num_features = (mfcc - 1) * 3 if deltas else mfcc - 1

    (x_train, y_train), (x_test, y_test) = \
        load_speakers_data_authentication(num_frames=frames, num_mfcc=mfcc, use_deltas=deltas,
                                          num_registered_male=num_registered_male,
                                          num_registered_female=num_registered_female,
                                          num_unregistered_male=num_unregistered_male,
                                          num_unregistered_female=num_unregistered_female,
                                          num_train_templates=train_templates, num_test_templates=num_test_templates)

    save_data_to_files(x_train, 'data/x_train.npy', y_train, 'data/y_train.npy', x_test, 'data/x_test.npy', y_test,
                       'data/y_test.npy')

    # (x_train, y_train), (x_test, y_test) = load_data_from_files('data/x_train.npy', 'data/y_train.npy',
    #                                                             'data/x_test.npy', 'data/y_test.npy')

    x_train = x_train.reshape(x_train.shape[0], frames, num_features, 1)
    x_test = x_test.reshape(x_test.shape[0], frames, num_features, 1)

    return cnn_models.get_second_model(input_shape=(frames, num_features, 1)), (x_train, y_train), (x_test, y_test)


def find_best_template_params(epochs):
    mfccs = [13, 22, 31]
    train_frames = 1200
    frames = [25, 50, 100]
    deltas = [False, True]

    results = []
    for mfcc in mfccs:
        for delta in deltas:
            for frame in frames:
                model, (x, y), _ = get_model(mfcc, delta, frame, train_frames // frame)
                print('mfcc = {0}, use_deltas = {1}, frames = {2}'.format(mfcc, delta, frame))
                results.append((cnn_models.k_fold_cross_val_score_f1(x, y, lambda: model, epochs), mfcc, delta, frame))
                # cnn_models.grid_search(x_train, y_train, x_test, y_test, lambda: network_model)

    print("Results:")
    t = PrettyTable(['mfcc', 'use_deltas', 'frames', 'F1-score', 'F1-score std'])
    for result in results:
        t.add_row([result[1], result[2], result[3], round(result[0].mean(), 4), round(result[0].std(), 4)])
    print(t)

    best_f1 = (numpy.array([0.0]), 0, 0, 0)
    for result in results:
        if result[0].mean() > best_f1[0].mean():
            best_f1 = (result[0], result[1], result[2], result[3])

    print('Best F1-score = {0}; mfcc = {1}, use_deltas = {2}, frames = {3}'.
          format(round(best_f1[0].mean(), 4), best_f1[1], best_f1[2], best_f1[3]))


if __name__ == '__main__':
    find_best_template_params(epochs=20)
