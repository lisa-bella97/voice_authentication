from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Загружаем данные x_train и x_test содержат двухмерный массив с изображение цифр
    # x_test, y_test массив с проверочными данными сети.
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # 6000 изображений в обучающей и 1000 в тестируемой выборке
    x_train = x_train[:6000]
    y_train = y_train[:6000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    # Трансформируем из двухмерного массива в трех мерный(28х28х1 канал)
    x_train = x_train.reshape(6000, 28, 28, 1)
    x_test = x_test.reshape(1000, 28, 28, 1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # создание модели
    model = Sequential()
    # Добавляем слой
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    # Второй сверточный слой
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # Создаем вектор для полносвязной сети.
    model.add(Flatten())
    # Создадим однослойный перцептрон
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2)
    print(history.history)

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
