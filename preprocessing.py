from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X = X.copy()
            # Encode gender: male=0, female=1
            if "Gender" in X.columns:
                X["Gender"] = X["Gender"].map({"male": 0, "female": 1})
            X.fillna(0, inplace=True)
            return X

        except Exception as e:
            raise Exception(f"Preprocessing error: {e}")
