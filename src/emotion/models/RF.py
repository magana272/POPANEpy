
class EmotionRandomForest:
    def EmotionRandomForest(self):
        pass

    def __init__(self, n_estimators=100, random_state=42):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        from sklearn.metrics import accuracy_score
        y_pred = self.model.predict(X)
        return accuracy_score(y, y_pred)

    def feature_importances(self):
        return self.model.feature_importances_

    def save_model(self, filepath):
        import joblib
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        import joblib
        self.model = joblib.load(filepath)
