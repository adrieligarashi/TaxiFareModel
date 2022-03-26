from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipe = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        '''returns a pipelined model'''
        self.dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                                   ('stdscaler', StandardScaler())])
        self.time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        self.preproc_pipe = ColumnTransformer([('distance', self.dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', self.time_pipe, ['pickup_datetime'])],
                                              remainder="drop")
        self.pipe = Pipeline([('preproc', self.preproc_pipe),
                              ('linear_model', LinearRegression())])
        return self.pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipe.fit(self.X, self.y)
        return self.pipe

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""

        self.y_pred = self.pipe.predict(X_test)
        self.rmse = compute_rmse(self.y_pred, y_test)
        print(self.rmse)
        return self.rmse



    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        self.client = MlflowClient()
        return self.client

    @memoized_property
    def mlflow_experiment_id(self):
        self.experiment_name = "[BR] [SP] [adrieligarashi] ex4 + 1"
        try:
            self.experiment_id = self.client.get_experiment_by_name(
                    self.experiment_name).experiment_id
            return self.client.create_experiment(self.experiment_name)
        except BaseException:
            return self.client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        print(
            f"experiment URL: https://mlflow.lewagon.co/#/experiments/{self.experiment_id}"
        )
        return self.client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.client.log_metric(self.mlflow_run.info.run_id, key, value)


    def save_model(self):
        joblib.dump(pipeline, 'model.joblib')


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    pipeline = Trainer(X_train, y_train)
    pipeline.run()
    RMSE = pipeline.evaluate(X_val, y_val)

    pipeline.mlflow_client
    pipeline.mlflow_experiment_id
    pipeline.mlflow_run
    pipeline.mlflow_log_param('model', 'linear')
    pipeline.mlflow_log_metric('RMSE', RMSE)

    pipeline.save_model()
