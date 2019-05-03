


def train_logistic_regression(X_data, Y_data, solver1='lbfgs'):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(solver=solver1)
    classifier.fit(X_data, Y_data)
    return classifier

def train_sequential_model(X_train, Y_train, X_test, Y_test):
    from keras.models import Sequential
    from keras import layers
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    trained_model = model.fit(X_train, Y_train, epochs=100, verbose=False, validation_data=(X_test, Y_test), batch_size=10)
    train_loss, train_accuracy = model.evaluate(X_train, Y_train, verbose=False)
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=False)
    print("Sequential Model Training Accuracy: {}".format(train_accuracy))
    print("Sequential Model Test Accuracy: {}".format(test_accuracy))
    return trained_model