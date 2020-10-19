import matplotlib.pyplot as plt
import sounddevice as sd
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, Convolution2D, BatchNormalization, \
    Activation, MaxPool2D
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from scipy.io.wavfile import write
from sklearn.metrics import roc_curve, auc
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.np_utils import to_categorical

from loaddata import load_data_from_files
from mfcc import get_mfcc_features_with_mean, get_mfcc_features
from utils import stem


def record(file):
    fs = 16000  # Sample rate
    seconds = 10  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(file, fs, myrecording)  # Save as WAV file


# Проверка средних значений коэффициентов
def check_mean_values():
    (liza1_mfcc, liza1_mfcc_mean), (liza1_deltas, liza1_deltas_mean), (
        liza1_deltas_deltas, liza1_deltas_deltas_mean) = get_mfcc_features_with_mean("samples/pausesDeleted_Liza_1.wav")
    (liza2_mfcc, liza2_mfcc_mean), (liza2_deltas, liza2_deltas_mean), (
        liza2_deltas_deltas, liza2_deltas_deltas_mean) = get_mfcc_features_with_mean("samples/pausesDeleted_Liza_2.wav")
    (test_mfcc, test_mfcc_mean), (test_deltas, test_deltas_mean), (
        test_deltas_deltas, test_deltas_deltas_mean) = get_mfcc_features_with_mean(
        "samples/ru_0075.wav")

    plt.subplot(3, 1, 1)
    stem(liza1_mfcc_mean, linefmt='r', markerfmt='ro')
    stem(liza2_mfcc_mean, linefmt='g', markerfmt='go')
    stem(test_mfcc_mean, linefmt='b', markerfmt='bo')
    plt.grid(True)
    plt.subplot(3, 1, 2)
    stem(liza1_deltas_mean, linefmt='r', markerfmt='ro')
    stem(liza2_deltas_mean, linefmt='g', markerfmt='go')
    stem(test_deltas_mean, linefmt='b', markerfmt='bo')
    plt.grid(True)
    plt.subplot(3, 1, 3)
    stem(liza1_deltas_deltas_mean, linefmt='r', markerfmt='ro')
    stem(liza2_deltas_deltas_mean, linefmt='g', markerfmt='go')
    stem(test_deltas_deltas_mean, linefmt='b', markerfmt='bo')
    plt.grid(True)
    plt.show()


# Проверка значений коэффициентов в определенных фреймах
def check_specific_values():
    num_coeff = 13
    liza1 = get_mfcc_features("samples/pausesDeleted_Liza_1.wav", num_coefficients=num_coeff)
    liza2 = get_mfcc_features("samples/pausesDeleted_Liza_2.wav")
    test = get_mfcc_features("samples/pausesDeleted_Liza_2.wav")

    # Проверка значений коэффициентов в каких-либо фреймах
    plt.subplot(3, 1, 1)
    stem(liza1[0, :num_coeff - 1], linefmt='r', markerfmt='ro')
    stem(liza1[1, :num_coeff - 1], linefmt='b', markerfmt='bo')
    stem(liza1[2, :num_coeff - 1], linefmt='y', markerfmt='yo')
    plt.grid(True)
    plt.subplot(3, 1, 2)
    stem(liza2[0, :num_coeff - 1], linefmt='r', markerfmt='ro')
    stem(liza2[1, :num_coeff - 1], linefmt='b', markerfmt='bo')
    stem(liza2[2, :num_coeff - 1], linefmt='y', markerfmt='yo')
    plt.grid(True)
    plt.subplot(3, 1, 3)
    stem(test[0, :num_coeff - 1], linefmt='r', markerfmt='ro')
    stem(test[1, :num_coeff - 1], linefmt='b', markerfmt='bo')
    stem(test[2, :num_coeff - 1], linefmt='y', markerfmt='yo')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # record("example.wav")

    # Количество фреймов в одном шаблоне
    num_frames = 100
    # Количество MFCC
    num_mfcc = 13
    # Количество коффициентов в одном фрейме = (кол-во MFCC)*3
    num_features = (num_mfcc - 1) * 3

    # (x_train, y_train), (x_test, y_test) = load_data()

    (x_train, y_train), (x_test, y_test) = load_data_from_files('data/x_train.npy', 'data/y_train.npy',
                                                                'data/x_test.npy', 'data/y_test.npy')

    # Трансформируем из двумерного массива в трехмерный (num_frames х num_features х 1 канал)
    x_train = x_train.reshape(x_train.shape[0], num_frames, num_features, 1)
    x_test = x_test.reshape(x_test.shape[0], num_frames, num_features, 1)

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_train)

    # создание модели
    # model = Sequential()

    # model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(num_frames, num_features, 1)))
    # model.add(Conv2D(64, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.25))
    # model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    # model.add(Conv2D(64, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.5))
    # model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    # model.add(Conv2D(128, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    inp = Input(shape=(num_frames, num_features, 1))
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

    model = Model(inputs=inp, outputs=out)

    # # Добавляем слой
    # model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(num_frames, num_features, 1)))
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.25))
    # # Второй сверточный слой
    # model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.5))
    # # Создаем вектор для полносвязной сети.
    # model.add(Flatten())
    # # Создадим однослойный перцептрон
    # model.add(Dense(1, activation='sigmoid'))

    plot_model(model, show_shapes=True)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2)
    print(history.history)

    # Plot training & validation accuracy values
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

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Потери при тестировании: ', score[0])
    print('Точность при тестировании:', score[1])

    y_pred = model.predict(x_test)
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
