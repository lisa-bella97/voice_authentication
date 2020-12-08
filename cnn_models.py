import numpy
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from tensorflow.python.keras.regularizers import l2
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
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

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


def get_eighth_model(input_shape, num_classes=None):
    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.25))

    if num_classes is None:
        model.add(Dense(2, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes * 2, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.25))
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001)))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def get_ninth_model(input_shape, num_classes=None):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
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
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    return results


def grid_search(x, y, x_test, y_test, get_model):
    seed = 7
    numpy.random.seed(seed)
    estimator = KerasClassifier(build_fn=get_model)
    epochs = [2]
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
    print()
