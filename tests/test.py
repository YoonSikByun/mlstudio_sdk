#################################################
# Set the Tracking Server URI (if not using a Databricks Managed MLflow Tracking Server)
#################################################
import mlflow
import os
from mlstudio_sdk.config import Config
from mlstudio_sdk.mlflow.api import create_experiment_if_not_exists, get_registered_model

config = Config()
print(config.get_mlflow_tracking_uri())
mlflow.set_tracking_uri(uri=config.get_mlflow_tracking_uri())
mlflow.end_run()


#################################################
# Train a model and prepare metadata for logging
#################################################
import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)


#################################################
# Set our tracking server uri for logging
#################################################
experiment_name = 'MLflow Quickstart2'

# registered_model_name = 'tracking-quickstart2'
# registered_model_name = 'test1_model1'

# Create a new MLflow Experiment
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
except AttributeError:
    raise Exception(f'does not exist experiment : {experiment_name}')

experiment = mlflow.set_experiment(experiment_name)
# update_artifact_location(experiment_id=experiment.experiment_id, artifact_location='/opt/mlflow/mlruns')

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        # registered_model_name=registered_model_name,
    )

#################################################
# Load the model back for predictions as a generic Python Function model
#################################################
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

result[:4]