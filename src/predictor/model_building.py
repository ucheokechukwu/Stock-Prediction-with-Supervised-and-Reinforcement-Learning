def eval (y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, accuracy_score

    import numpy as np
    y_pred = np.squeeze(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred)
    cf_matrix = confusion_matrix(y_true, y_pred)

    return {'accuracy':accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1}, \
            {'classification_report': clf_report,
            'confusion_matrix': cf_matrix}
            
def model_evaluation(model, modelname, filepath, X_test, y_test, metrics_log=metrics_log):
    """loads the saved best performing model
    and calls the eval function to evaluate it
    """
    model.load_weights(filepath)
    display("shape of x-te:", X_test.shape)
    y_pred = np.squeeze(model.predict(X_test)).argmax(axis=1)
    display("shape of y_pred:", y_pred.shape)
    scores = eval(y_test, y_pred)
    metrics_log[modelname] = scores[0]
    display(metrics_log)
    return scores


def save_best(model):
    """
    saves the best epoch performing weights into a filepath
    that is based on the name of the model
    """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath='output/'+model,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='auto',
        save_best_only=True)

# function to train test split the X and y
def train_test_splits(data, test_split=0.2):
    """
    Input: X and y
    Output: X_train, X_test, y_train, y_test in that order
    """

    data = pd.concat([data[:'2021-10'], data['2022-06':]])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    tmp = data.copy()
    X = tmp.drop(columns=['Forecast'])
    y = tmp['Forecast'].values
    X = scaler.fit_transform(X)

    # save this scaler
    import pickle
    # pickle.dump(scaler, open('output/model/scaler', 'wb'))


    split = round(len(X) * (1-test_split))
    # returns in order: X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    display (X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test,


def train_model(model, data, model_name, epochs=1000):
    # definining checkpoint
    early_stop = EarlyStopping(monitor='loss', patience=10)
    # train test split
    X_train, X_test, y_train, y_test = train_test_splits(data)

    history = model.fit(X_train,
           y_train,
           epochs=epochs,
           verbose=1,
           batch_size=128,
           validation_data=(X_test, y_test),
           callbacks=[early_stop, save_best(model_name)]
           ) # create ModelCheckpoint callback to save best model

    return history, 'output/'+model_name, X_test, y_test