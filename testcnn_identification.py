from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import numpy
import tensorflow
from keras import models
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, Convolution2D, BatchNormalization, \
    Activation, MaxPool2D
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.models import Model
import keras.backend as K
from tensorflow.keras import datasets, layers
from tensorflow.python.keras.utils.np_utils import to_categorical

from loaddata import load_data_from_files, load_speakers_data, save_data_to_files, load_templates_from_directories


def get_first_model(input_shape, num_classes=None):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())

    if num_classes is None:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    return model


def get_second_model(input_shape, num_classes=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    if num_classes is None:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    return model


def get_third_model(input_shape, num_classes=None):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=4, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same'))

    model.add(Conv2D(32 * 2, kernel_size=(4, 10), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same'))

    model.add(Conv2D(32 * 3, kernel_size=(4, 10), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same'))

    model.add(Conv2D(32 * 3, kernel_size=(4, 10), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())

    if num_classes is None:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    return model


def get_fourth_model(input_shape, num_classes=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))

    if num_classes is None:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    return model


def get_fifth_model(input_shape, num_classes=None):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, input_shape=(num_frames, num_features, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(256, kernel_size=3, input_shape=(num_frames, num_features, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())

    if num_classes is None:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    return model


def get_sixth_model(input_shape, num_classes=None):
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    if num_classes is None:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    return model


def get_seventh_model(input_shape, num_classes=None):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    if num_classes is None:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    return model


# def cross_val_score(X, Y, input_shape):
#     # fix random seed for reproducibility
#     seed = 7
#     numpy.random.seed(seed)
#     # define 10-fold cross validation test harness
#     kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#     cvscores = []
#     for train, test in kfold.split(X, Y):
#         model = get_sixth_model(input_shape=input_shape)
#         model.compile(optimizer='adam', loss='binary_crossentropy',
#                       metrics=['accuracy', count_f1, count_precision, count_recall])
#         model.fit(X[train], Y[train], epochs=40, verbose=0)
#         loss, accuracy, f1_score, precision, recall = model.evaluate(X[test], Y[test], verbose=0)
#         print("%s: %.2f%%" % ("loss", loss * 100))
#         print("%s: %.2f%%" % ("accuracy", accuracy * 100))
#         print("%s: %.2f%%" % ("f1_score", f1_score))
#         print("%s: %.2f%%" % ("precision", precision * 100))
#         print("%s: %.2f%%" % ("recall", recall))
#         cvscores.append(accuracy * 100)
#     print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


if __name__ == '__main__':
    # Количество фреймов в одном шаблоне
    num_frames = 50
    # Количество шаблонов для обучения (на одного диктора)
    num_train_templates = 25
    # Количество шаблонов для тестирования (на одного диктора)
    num_test_templates = num_train_templates // 5
    # Количество MFCC
    num_mfcc = 25
    # Будут ли считаться первые и вторые производные
    use_deltas = False
    # Количество коэффициентов в одном фрейме
    num_features = (num_mfcc - 1) * 3 if use_deltas else num_mfcc - 1
    # Количество эпох обучения
    num_epochs = 40
    # Количество зарегистрированных дикторов
    num_speakers = 50
    num_female = 9
    num_male = num_speakers - num_female

    (x_train, y_train), (x_test, y_test) = load_speakers_data(num_frames=num_frames, num_mfcc=num_mfcc,
                                                              use_deltas=use_deltas, num_registered_male=43,
                                                              num_registered_female=7, num_unregistered_male=44,
                                                              num_unregistered_female=6, num_train_files=3,
                                                              num_test_files=1)

    # speakers = sorted([join("speakers", "russian", "male", d) for d in listdir(join("speakers", "russian", "male"))])[
    #            :num_male]
    # speakers.extend(
    #     sorted([join("speakers", "russian", "female", d) for d in listdir(join("speakers", "russian", "female"))])[
    #     :num_female])
    #
    # (x_train, y_train), (x_test, y_test) = \
    #     load_templates_from_directories(directories=speakers, num_frames=num_frames, num_mfcc=num_mfcc,
    #                                     use_deltas=use_deltas, num_train_templates=num_train_templates,
    #                                     num_test_templates=num_test_templates)

    save_data_to_files(x_train, y_train, x_test, y_test)

    # (x_train, y_train), (x_test, y_test) = load_data_from_files('data/x_train.npy', 'data/y_train.npy',
    #                                                             'data/x_test.npy', 'data/y_test.npy')

    x_train = x_train.reshape(x_train.shape[0], num_frames, num_features, 1)
    x_test = x_test.reshape(x_test.shape[0], num_frames, num_features, 1)

    # y_train_categorical = to_categorical(y_train)
    # y_test_categorical = to_categorical(y_test)

    # cross_val_score(x_train, y_train)

    network_model = get_seventh_model(input_shape=(num_frames, num_features, 1), num_classes=num_speakers)
    # plot_model(network_model, show_shapes=True)
    network_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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

    # yhat_classes = (network_model.predict(x_test) > 0.5).astype("int32")
    yhat_classes = network_model.predict_classes(x_test, verbose=0)
    # reduce to 1d array
    # yhat_classes = yhat_classes[:, 0]

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes, average='micro')
    print('F1 score: %.3f' % f1)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes, average='micro')
    print('Precision: %.3f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes, average='micro')
    print('Recall: %.3f' % recall)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %.3f' % accuracy)

    score = network_model.evaluate(x_test, y_test_categorical, verbose=0)
    print('Потери при тестировании: %.3f' % score[0])
    print('Точность при тестировании: %.3f' % score[1])

    # y_pred = network_model.predict(x_test)
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred.ravel())
    # auc_keras = auc(fpr, tpr)
    #
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr, tpr, label='ROC area = {:.3f}'.format(auc_keras))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.grid()
    # plt.show()
