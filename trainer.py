import joblib
import os
from sklearn.model_selection import train_test_split

class ModelTrainer:

    def train_and_save(self, pipeline, X, y, model_path):
        try:
            # Create artifacts directory if it doesn't exist
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            pipeline.fit(X_train, y_train)

            # Save trained model
            joblib.dump(pipeline, model_path)

            return pipeline, X_test, y_test

        except Exception as e:
            raise Exception(f"Training error: {e}")
