from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    return clf

def predict_accuracy(model, X_test, y_test):
    accuracy = (sum(model.predict(X_test) == y_test) / len(y_test)) * 100
    return accuracy