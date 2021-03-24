from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy
import tensorflow
from keras import models
from sklearn.metrics import roc_curve, auc, classification_report

import cnn_models
from loaddata import load_speakers_data_authentication, save_data_to_files, load_data_from_files


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

    x_train = x_train.reshape(x_train.shape[0], frames, num_features, 1)
    x_test = x_test.reshape(x_test.shape[0], frames, num_features, 1)

    # save_data_to_files(x_train, 'data/test_far_frr/x_train.npy', y_train, 'data/test_far_frr/y_train.npy', x_test,
    #                    'data/test_far_frr/x_test.npy', y_test, 'data/test_far_frr/y_test.npy')

    # (x_train, y_train), (x_test, y_test) = \
    #     load_data_from_files('data/test_far_frr/x_train.npy', 'data/test_far_frr/y_train.npy',
    #                          'data/test_far_frr/x_test.npy', 'data/test_far_frr/y_test.npy')

    return cnn_models.get_second_model(input_shape=(frames, num_features, 1)), (x_train, y_train), (x_test, y_test)


def find_best_template_params(epochs):
    # train_frames = frames * num_train_templates; train_frames <= 1500 (из-за датасета); train : test = 60 : 40
    train_frames = [300, 900, 1500]
    mfccs = [13, 21, 31]
    frames = [25, 50, 100]
    deltas = [False, True]

    results = []

    for mfcc in mfccs:
        for delta in deltas:
            for frame in frames:
                for train_frame in train_frames:
                    model, (x_train, y_train), (x_test, y_test) = get_model(mfcc, delta, frame, train_frame // frame)
                    x = numpy.concatenate((x_train, x_test), axis=0)
                    y = numpy.concatenate((y_train, y_test), axis=0)
                    print('mfcc = {0}, use_deltas = {1}, frames = {2}, train_frames = {3}'.format(mfcc, delta, frame,
                                                                                                  train_frame))

                    k_fold_results, fars, frrs = cnn_models.k_fold_cross_val_score(x, y, lambda: model, epochs)
                    results.append((k_fold_results, fars, frrs, mfcc, delta, frame, train_frame))

    print("Results:")
    t = PrettyTable(['mfcc', 'use_deltas', 'frames', 'train_frames', 'accuracy', 'FAR', 'FRR'])
    for result in results:
        t.add_row([result[3], result[4], result[5], result[6], result[0].mean(), result[1].mean(), result[2].mean()])
    print(t)

    best_accuracy = (numpy.array([0.0]), 0, 0, 0, 0)
    best_far = (numpy.array([1.0]), 0, 0, 0, 0)
    best_frr = (numpy.array([1.0]), 0, 0, 0, 0)
    for result in results:
        if result[0].mean() > best_accuracy[0].mean():
            best_accuracy = (result[0], result[3], result[4], result[5])
        if result[1].mean() < best_far[0].mean():
            best_far = (result[1], result[3], result[4], result[5])
        if result[2].mean() < best_frr[0].mean():
            best_frr = (result[2], result[3], result[4], result[5])

    print('Best accuracy = {0}; mfcc = {1}, use_deltas = {2}, frames = {3}, train_frames = {4}'.
          format(best_accuracy[0].mean(), best_accuracy[1], best_accuracy[2], best_accuracy[3], best_accuracy[4]))
    print('Best FAR = {0}; mfcc = {1}, use_deltas = {2}, frames = {3}, train_frames = {4}'.
          format(best_far[0].mean(), best_far[1], best_far[2], best_far[3], best_far[4]))
    print('Best FRR = {0}; mfcc = {1}, use_deltas = {2}, frames = {3}, train_frames = {4}'.
          format(best_frr[0].mean(), best_frr[1], best_frr[2], best_frr[3], best_frr[4]))


if __name__ == '__main__':
    # Количество MFCC
    num_mfcc = 31
    # Будут ли считаться первые и вторые производные
    use_deltas = True
    # Количество эпох обучения
    num_epochs = 20
    # Количество фреймов в одном шаблоне
    num_frames = 100
    # Количество шаблонов для обучения (на одного диктора)
    num_train_templates = 12

    find_best_template_params(num_epochs)

    network_model, (x_train, y_train), (x_test, y_test) = get_model(num_mfcc, use_deltas, num_frames,
                                                                    num_train_templates)
    # plot_model(network_model)

    history = network_model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test))
    print(history.history)

    network_model.save('far_frr_authentication_model')

    # plt.plot(list(range(1, num_epochs + 1)), history.history['accuracy'])
    # plt.plot(list(range(1, num_epochs + 1)), history.history['val_accuracy'])
    # plt.xlim([0, num_epochs])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.grid()
    # plt.show()
    #
    # plt.plot(list(range(1, num_epochs + 1)), history.history['loss'])
    # plt.plot(list(range(1, num_epochs + 1)), history.history['val_loss'])
    # plt.xlim([0, num_epochs])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.grid()
    # plt.show()

    # network_model = models.load_model('far_frr_authentication_model')
    # (x_train, y_train), (x_test, y_test) = \
    #     load_data_from_files('data/test_far_frr/x_train.npy', 'data/test_far_frr/y_train.npy',
    #                          'data/test_far_frr/x_test.npy', 'data/test_far_frr/y_test.npy')

    score = network_model.evaluate(x_test, y_test, verbose=0)
    print('Потери при тестировании: %.2f' % score[0])
    print('Точность при тестировании: %.2f' % score[1])

    y_pred = network_model.predict(x_test)
    print(classification_report(y_test, y_pred.round()))

    m = tensorflow.keras.metrics.FalsePositives()
    m.update_state(y_test, y_pred.ravel())
    print("FAR = ", m.result().numpy() / y_test.shape[0])

    m.reset_states()
    m = tensorflow.keras.metrics.FalseNegatives()
    m.update_state(y_test, y_pred.ravel())
    print("FRR = ", m.result().numpy() / y_test.shape[0])

    fpr, tpr, thresholds = roc_curve(y_test, y_pred.ravel())
    auc_keras = auc(fpr, tpr)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='ROC area = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
