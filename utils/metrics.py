from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype('int32')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
