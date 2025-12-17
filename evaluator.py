from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

class Evaluator:

    def evaluate(self, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            return {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2 Score": r2
            }

        except Exception as e:
            raise Exception(f"Evaluation error: {e}")
