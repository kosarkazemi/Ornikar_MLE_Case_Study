from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    # Your model training logic here
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model
