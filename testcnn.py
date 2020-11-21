import matplotlib.pyplot as plt
import numpy
from keras import models
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, Convolution2D, BatchNormalization, \
    Activation, MaxPool2D
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.models import Model
import keras.backend as K

from loaddata import load_data_from_files, load_speakers_data, save_data_to_files


def get_first_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


def get_second_model(input_shape):
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
    model.add(Dense(1, activation='sigmoid'))

    return model


def get_third_model(input_shape):
    inp = Input(shape=input_shape)
    x = Convolution2D(32, (4, 4), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32 * 2, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32 * 3, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32 * 3, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model(inputs=inp, outputs=out)


def get_fourth_model(input_shape):
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
    model.add(Dense(1, activation='sigmoid'))

    return model


def get_fifth_model(input_shape):
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
    model.add(Dense(1, activation='sigmoid'))

    return model


def get_sixth_model(input_shape):
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
    model.add(Dense(1, activation='sigmoid'))

    return model


def get_seventh_model(input_shape):
    inp = Input(shape=input_shape)
    conv_1 = Convolution2D(32, kernel_size=3, padding='same', activation='relu')(inp)
    conv_2 = Convolution2D(32, kernel_size=3, padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D()(conv_2)
    drop_1 = Dropout(0.25)(pool_1)

    conv_3 = Convolution2D(64, kernel_size=3, padding='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(64, kernel_size=3, padding='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D()(conv_4)
    drop_2 = Dropout(0.25)(pool_2)

    flat = Flatten()(drop_2)
    hidden = Dense(512, activation='relu')(flat)
    drop_3 = Dropout(0.5)(hidden)
    out = Dense(1, activation='sigmoid')(drop_3)

    return Model(inputs=inp, outputs=out)


def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def cross_val_score(X, Y):
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):
        model = get_sixth_model(input_shape=(num_frames, num_features, 1))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
        model.fit(X[train], Y[train], epochs=40, verbose=0)
        loss, accuracy, f1_score, precision, recall = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % ("loss", loss * 100))
        print("%s: %.2f%%" % ("accuracy", accuracy * 100))
        print("%s: %.2f%%" % ("f1_score", f1_score))
        print("%s: %.2f%%" % ("precision", precision * 100))
        print("%s: %.2f%%" % ("recall", recall))
        cvscores.append(accuracy * 100)
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


if __name__ == '__main__':
    # Количество фреймов в одном шаблоне
    num_frames = 100
    # Количество MFCC
    num_mfcc = 13
    # Количество коффициентов в одном фрейме = (кол-во MFCC)*3
    num_features = (num_mfcc - 1) * 3

    (x_train, y_train), (x_test, y_test) = load_data_from_files('data/x_train.npy', 'data/y_train.npy',
                                                                'data/x_test.npy', 'data/y_test.npy')

    # (x_train, y_train), (x_test, y_test) = load_speakers_data(num_frames=num_frames, num_mfcc=num_mfcc,
    #                                                           num_registered=5, num_unregistered=10, num_train_files=3,
    #                                                           num_test_files=1)
    # save_data_to_files(x_train, y_train, x_test, y_test)

    x_train = x_train.reshape(x_train.shape[0], num_frames, num_features, 1)
    x_test = x_test.reshape(x_test.shape[0], num_frames, num_features, 1)

    # cross_val_score(x_train, y_train)

    network_model = get_sixth_model(input_shape=(num_frames, num_features, 1))
    plot_model(network_model, show_shapes=True)
    network_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
    history = network_model.fit(x_train, y_train, epochs=40)
    print(history.history)

    network_model.save('model')
    # network_model = models.load_model('model')

    # scores = cross_val_score(network_model, x_train, y_train, cv=5, scoring='f1_macro')
    # print(scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    loss, accuracy, f1_score, precision, recall = network_model.evaluate(x_test, y_test)
    print("%s: %.2f%%" % ("loss", loss * 100))
    print("%s: %.2f%%" % ("accuracy", accuracy * 100))
    print("%s: %.2f%%" % ("f1_score", f1_score))
    print("%s: %.2f%%" % ("precision", precision * 100))
    print("%s: %.2f%%" % ("recall", recall))

    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.grid()
    # plt.show()
    # #
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.grid()
    # plt.show()
    #
    # score = network_model.evaluate(x_test, y_test, verbose=0)
    # print('Потери при тестировании: ', score[0])
    # print('Точность при тестировании:', score[1])
    #
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
