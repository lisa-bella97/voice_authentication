import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.python.keras.utils.np_utils import to_categorical

import cnn_models
from loaddata import load_speakers_data_identification, save_data_to_files, load_data_from_files


def get_model(mfcc=13, deltas=True, frames=25, train_templates=48):
    # Количество шаблонов для тестирования (на одного диктора) (60%:40%)
    num_test_templates = int(train_templates // 1.5)
    # Количество коэффициентов в одном фрейме
    num_features = (mfcc - 1) * 3 if deltas else mfcc - 1
    # Количество зарегистрированных дикторов
    num_speakers = 50

    (x_train, y_train), (x_test, y_test) = \
        load_speakers_data_identification(num_frames=frames, num_mfcc=mfcc, use_deltas=deltas,
                                          num_speakers=num_speakers, num_female=9, num_train_templates=train_templates,
                                          num_test_templates=num_test_templates)

    save_data_to_files(x_train, 'data/x_train.npy', y_train, 'data/y_train.npy', x_test, 'data/x_test.npy', y_test,
                       'data/y_test.npy')

    # (x_train, y_train), (x_test, y_test) = load_data_from_files('data/x_train.npy', 'data/y_train.npy',
    #                                                             'data/x_test.npy', 'data/y_test.npy')

    x_train = x_train.reshape(x_train.shape[0], frames, num_features, 1)
    x_test = x_test.reshape(x_test.shape[0], frames, num_features, 1)

    return cnn_models.get_third_model(input_shape=(frames, num_features, 1), num_classes=num_speakers), \
           (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    # Количество MFCC
    num_mfcc = 31
    # Будут ли считаться первые и вторые производные
    use_deltas = False
    # Количество эпох обучения
    num_epochs = 40
    # Количество фреймов в одном шаблоне
    num_frames = 25
    # Количество шаблонов для обучения (на одного диктора)
    num_train_templates = 48

    network_model, (x_train, y_train), (x_test, y_test) = get_model(num_mfcc, use_deltas, num_frames,
                                                                    num_train_templates)
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)
    # plot_model(network_model)
    history = network_model.fit(x_train, y_train_categorical, epochs=num_epochs,
                                validation_data=(x_test, y_test_categorical))
    print(history.history)

    network_model.save('identification_model')

    plt.plot(list(range(1, num_epochs + 1)), history.history['accuracy'])
    plt.plot(list(range(1, num_epochs + 1)), history.history['val_accuracy'])
    plt.xlim([0, num_epochs])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

    plt.plot(list(range(1, num_epochs + 1)), history.history['loss'])
    plt.plot(list(range(1, num_epochs + 1)), history.history['val_loss'])
    plt.xlim([0, num_epochs])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()

    # network_model = models.load_model('identification_model')

    score = network_model.evaluate(x_test, y_test_categorical, verbose=0)
    print('Потери при тестировании: %.2f' % score[0])
    print('Точность при тестировании: %.2f' % score[1])

    yhat_classes = network_model.predict_classes(x_test, verbose=0)
    print(classification_report(y_test, yhat_classes))
