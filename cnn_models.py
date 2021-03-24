import numpy
import tensorflow
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


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
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def get_second_model(input_shape, num_classes=None):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    if num_classes is None:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def get_third_model(input_shape, num_classes=None):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=4, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same'))

    model.add(Conv2D(32 * 2, kernel_size=(4, 10), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same'))

    model.add(Conv2D(32 * 3, kernel_size=(4, 10), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same'))

    model.add(Conv2D(32 * 3, kernel_size=(4, 10), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())

    if num_classes is None:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

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
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def get_fifth_model(input_shape, num_classes=None):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(256, kernel_size=3))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())

    if num_classes is None:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

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
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

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
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def k_fold_cross_val_score(x, y, get_model, epochs):
    estimator = KerasClassifier(build_fn=get_model, epochs=epochs)
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    results = cross_val_score(estimator, x, y, cv=kfold)
    print("Average score = %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    fars, frrs = [], []
    for train, test in kfold.split(x, y):
        y_pred = get_model().predict(x[test])
        m = tensorflow.keras.metrics.FalsePositives()
        m.update_state(y[test], y_pred.ravel())
        far = m.result().numpy() / y[test].shape[0]
        print("FAR = ", far)
        fars.append(far)

        m.reset_states()
        m = tensorflow.keras.metrics.FalseNegatives()
        m.update_state(y[test], y_pred.ravel())
        frr = m.result().numpy() / y[test].shape[0]
        print("FRR = ", frr)
        frrs.append(frr)

    fars = numpy.array(fars)
    frrs = numpy.array(frrs)
    print("Average FAR = %.4f (%.4f)" % (fars.mean(), fars.std()))
    print("Average FRR = %.4f (%.4f)" % (frrs.mean(), frrs.std()))

    return results, fars, frrs


def grid_search(x, y, x_test, y_test, get_model):
    seed = 7
    numpy.random.seed(seed)
    estimator = KerasClassifier(build_fn=get_model)
    epochs = [3]
    param_grid = dict(epochs=epochs)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid)
    grid_res = grid.fit(x, y)
    print("Best: %f using %s" % (grid_res.best_score_, grid_res.best_params_))

    means = grid_res.cv_results_['mean_test_score']
    stds = grid_res.cv_results_['std_test_score']
    params = grid_res.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    print("Detailed classification report:")
    y_true, y_pred = y_test, grid.predict(x_test)
    print(classification_report(y_true, y_pred))
    m = tensorflow.keras.metrics.FalsePositives()
    m.update_state(y_test, y_pred.ravel())
    print("FAR = ", m.result().numpy() / y_test.shape[0])
    m.reset_states()
    m = tensorflow.keras.metrics.FalseNegatives()
    m.update_state(y_test, y_pred.ravel())
    print("FRR = ", m.result().numpy() / y_test.shape[0])
    print()
