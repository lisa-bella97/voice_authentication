from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import classification_report
from tensorflow.python.keras.utils.np_utils import to_categorical

import cnn_models
from loaddata import save_data_to_files, load_templates_from_directories


def get_model(mfcc=13, deltas=True, frames=25, train_templates=48):
    # Количество шаблонов для тестирования (на одного диктора) (60%:40%)
    num_test_templates = int(train_templates // 1.5)
    # Количество коэффициентов в одном фрейме
    num_features = (mfcc - 1) * 3 if deltas else mfcc - 1
    # Количество зарегистрированных дикторов
    num_speakers = 50
    num_female = 9
    num_male = num_speakers - num_female

    speakers = sorted([join("speakers", "russian", "male", d) for d in listdir(join("speakers", "russian", "male"))])[
               :num_male]
    speakers.extend(
        sorted([join("speakers", "russian", "female", d) for d in listdir(join("speakers", "russian", "female"))])[
        :num_female])

    (x_train, y_train), (x_test, y_test) = \
        load_templates_from_directories(directories=speakers, num_frames=frames, num_mfcc=mfcc, use_deltas=deltas,
                                        num_train_templates=train_templates, num_test_templates=num_test_templates)

    save_data_to_files(x_train, y_train, x_test, y_test)

    # (x_train, y_train), (x_test, y_test) = load_data_from_files('data/x_train.npy', 'data/y_train.npy',
    #                                                             'data/x_test.npy', 'data/y_test.npy')

    x_train = x_train.reshape(x_train.shape[0], frames, num_features, 1)
    x_test = x_test.reshape(x_test.shape[0], frames, num_features, 1)

    return cnn_models.get_third_model(input_shape=(frames, num_features, 1), num_classes=num_speakers), \
           (x_train, y_train), (x_test, y_test)


def find_best_template_params(epochs):
    mfccs = [13, 21, 31]
    train_frames = 1200
    frames = [25, 50, 100]
    deltas = [False, True]

    results = []
    for mfcc in mfccs:
        for delta in deltas:
            for frame in frames:
                model, (x, y), _ = get_model(mfcc, delta, frame, train_frames // frame)
                print('mfcc = {0}, use_deltas = {1}, frames = {2}'.format(mfcc, delta, frame))
                results.append((cnn_models.k_fold_cross_val_score(x, y, lambda: model, epochs), mfcc, delta, frame))
                # cnn_models.grid_search(x_train, y_train, x_test, y_test, lambda: network_model)

    best = (numpy.array([0]), 0, 0, 0)
    for result in results:
        if result[0].mean() > best[0].mean():
            best = result
    print('Best result = {0}; mfcc = {1}, use_deltas = {2}, frames = {3}'.format(best[0].mean(), best[1], best[2],
                                                                                 best[3]))


if __name__ == '__main__':
    # Количество MFCC
    num_mfcc = 13
    # Будут ли считаться первые и вторые производные
    use_deltas = True
    # Количество эпох обучения
    num_epochs = 40
    # Количество фреймов в одном шаблоне
    num_frames = 25
    # Количество шаблонов для обучения (на одного диктора)
    num_train_templates = 48

    # find_best_template_params(num_epochs)

    network_model, (x_train, y_train), (x_test, y_test) = get_model(num_mfcc, use_deltas, num_frames,
                                                                    num_train_templates)
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)
    # plot_model(network_model)
    history = network_model.fit(x_train, y_train_categorical, epochs=num_epochs,
                                validation_data=(x_test, y_test_categorical))
    print(history.history)

    network_model.save('model')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

    # network_model = models.load_model('model')

    score = network_model.evaluate(x_test, y_test_categorical, verbose=0)
    print('Потери при тестировании: %.2f' % score[0])
    print('Точность при тестировании: %.2f' % score[1])

    yhat_classes = network_model.predict_classes(x_test, verbose=0)
    print(classification_report(y_test, yhat_classes))
