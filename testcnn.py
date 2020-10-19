import matplotlib.pyplot as plt
from keras import models
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, Convolution2D, BatchNormalization, \
    Activation, MaxPool2D
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_curve, auc
from tensorflow.python.keras.models import Model

from loaddata import load_data_from_files


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


if __name__ == '__main__':
    # Количество фреймов в одном шаблоне
    num_frames = 100
    # Количество MFCC
    num_mfcc = 13
    # Количество коффициентов в одном фрейме = (кол-во MFCC)*3
    num_features = (num_mfcc - 1) * 3

    # (x_train, y_train), (x_test, y_test) = load_data()
    (x_train, y_train), (x_test, y_test) = load_data_from_files('data/x_train.npy', 'data/y_train.npy',
                                                                'data/x_test.npy', 'data/y_test.npy')

    x_train = x_train.reshape(x_train.shape[0], num_frames, num_features, 1)
    x_test = x_test.reshape(x_test.shape[0], num_frames, num_features, 1)

    network_model = get_seventh_model(input_shape=(num_frames, num_features, 1))
    plot_model(network_model, show_shapes=True)
    network_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = network_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=40)
    print(history.history)

    # network_model.save('model')
    # network_model = models.load_model('model')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()
    #
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

    score = network_model.evaluate(x_test, y_test, verbose=0)
    print('Потери при тестировании: ', score[0])
    print('Точность при тестировании:', score[1])

    y_pred = network_model.predict(x_test)
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
