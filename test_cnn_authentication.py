import matplotlib.pyplot as plt
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

    save_data_to_files(x_train, 'data/x_train.npy', y_train, 'data/y_train.npy', x_test, 'data/x_test.npy', y_test,
                       'data/y_test.npy')

    # (x_train, y_train), (x_test, y_test) = load_data_from_files('data/x_train.npy', 'data/y_train.npy',
    #                                                             'data/x_test.npy', 'data/y_test.npy')

    x_train = x_train.reshape(x_train.shape[0], frames, num_features, 1)
    x_test = x_test.reshape(x_test.shape[0], frames, num_features, 1)

    return cnn_models.get_second_model(input_shape=(frames, num_features, 1)), (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    # Количество MFCC
    num_mfcc = 31
    # Будут ли считаться первые и вторые производные
    use_deltas = False
    # Количество эпох обучения
    num_epochs = 40
    # Количество фреймов в одном шаблоне
    num_frames = 50
    # Количество шаблонов для обучения (на одного диктора)
    num_train_templates = 24

    network_model, (x_train, y_train), (x_test, y_test) = get_model(num_mfcc, use_deltas, num_frames,
                                                                    num_train_templates)
    # plot_model(network_model)

    history = network_model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test))
    print(history.history)

    network_model.save('authentication_model')

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

    # network_model = models.load_model('authentication_model')

    score = network_model.evaluate(x_test, y_test, verbose=0)
    print('Потери при тестировании: %.2f' % score[0])
    print('Точность при тестировании: %.2f' % score[1])

    y_pred = network_model.predict(x_test)
    print(classification_report(y_test, y_pred.round()))

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
