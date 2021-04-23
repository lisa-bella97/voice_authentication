import numpy
import tensorflow
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def k_fold_cross_val_score_accuracy(x, y, get_model, epochs):
    accuracy_estimator = KerasClassifier(build_fn=get_model, epochs=epochs)
    accuracy_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    accuracies = cross_val_score(accuracy_estimator, x, y, cv=accuracy_kfold)
    print("Average accuracy = %.2f%% (%.2f%%)" % (accuracies.mean() * 100, accuracies.std() * 100))
    return accuracies


def k_fold_cross_val_score_f1(x, y, get_model, epochs):
    f1_estimator = KerasClassifier(build_fn=get_model, epochs=epochs)
    f1_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    f1_scores = cross_val_score(f1_estimator, x, y, cv=f1_kfold, scoring='f1')
    print("Average F1-score = %.4f (%.4f)" % (f1_scores.mean(), f1_scores.std()))
    return f1_scores


def k_fold_cross_val_score_f1_micro(x, y, get_model, epochs):
    f1_estimator = KerasClassifier(build_fn=get_model, epochs=epochs)
    f1_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    f1_scores = cross_val_score(f1_estimator, x, y, cv=f1_kfold, scoring='f1_micro')
    print("Average F1-score = %.4f (%.4f)" % (f1_scores.mean(), f1_scores.std()))
    return f1_scores


def k_fold_cross_val_score_with_far_frr(x, y, get_model, epochs):
    estimator = KerasClassifier(build_fn=get_model, epochs=epochs)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
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
