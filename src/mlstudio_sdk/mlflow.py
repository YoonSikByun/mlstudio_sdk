import os
import mlflow
from mlflow import MlflowClient
from mlflow.server import get_app_client
import psycopg2
from mlstudio_sdk.config import Config
from mlflow.exceptions import MlflowException

config = Config()

def set_tracking_user_env(login_id='', login_pwd='') :
    if login_id:
        os.environ['MLFLOW_TRACKING_USERNAME'] = login_id
    if login_pwd:
        os.environ['MLFLOW_TRACKING_PASSWORD'] = login_pwd
  
# ML Flow 사용자를 추가한다.
def create_user(login_id, login_pwd, user_id, user_pwd) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    auth_client = get_app_client("basic-auth", tracking_uri=config.get_mlflow_tracking_uri())
    auth_client.create_user(username=user_id, password=user_pwd)

# ML Flow 사용자를 삭제한다.
def delete_user(login_id, login_pwd, user_id) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    auth_client = get_app_client("basic-auth", tracking_uri=config.get_mlflow_tracking_uri())
    auth_client.delete_user(username=user_id)

# ML Flow 사용자에게 Experiment 사용권한 부여한다.
def apply_experiment_permission(login_id, login_pwd, experiment_name, user_id, permission) :
    # Permission      |  Can read | Can update | Can delete | Can manage
    # READ               Yes          No           No            No
    # EDIT               Yes          Yes          No            No
    # MANAGE             Yes          Yes          Yes           Yes
    # NO_PERMISSIONS     No           No           No            No
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())
    auth_client = get_app_client("basic-auth", tracking_uri=config.get_mlflow_tracking_uri())

    experiment_details = client.get_experiment_by_name(experiment_name)

    if experiment_details :
        experiment_id = experiment_details.experiment_id
    else :
        raise Exception(f'{experiment_name} does not exist.')

    auth_client.create_experiment_permission(experiment_id=experiment_id, username=user_id, permission=permission)

# ML Flow 사용자에게 Experiment 부여된 사용권한 취소한다.
def cancel_experiment_permission(login_id, login_pwd, experiment_name, user_id) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)
    
    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())
    auth_client = get_app_client("basic-auth", tracking_uri=config.get_mlflow_tracking_uri())

    experiment_details = client.get_experiment_by_name(experiment_name)

    if experiment_details :
        experiment_id = experiment_details.experiment_id
    else :
        raise Exception(f'{experiment_name} does not exist.')

    auth_client.delete_experiment_permission(experiment_id=experiment_id, username=user_id)

def delete_experiment(login_id, login_pwd, experiment_name) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())
    experiment_details = client.get_experiment_by_name(experiment_name)
    
    if experiment_details :
        experiment_id = experiment_details.experiment_id
    else :
        raise Exception(f'{experiment_name} does not exist.')

    client.delete_experiment(experiment_id)

def create_experiment(login_id, login_pwd, name : str, tags : dict = {} ) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())
    experiment_id = client.create_experiment(
        name=name,
        artifact_location=config.get_mlflow_artifact_url(),
        tags=tags)

    return experiment_id

def create_experiment_if_not_exists(login_id, login_pwd, experiment_name : str, tags : dict = {} ) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        return experiment.experiment_id
    except AttributeError:
        # raise Exception(f'does not exist experiment : {experiment_name}')
        pass
    
    experiment_id = client.create_experiment(
        name=experiment_name,
        artifact_location=config.get_mlflow_artifact_url(),
        tags=tags)

    return experiment_id

# def create_registered_model_permission(registered_model_name, username, permission) :

def create_registered_model(login_id, login_pwd, name, desc : str = '', tags : dict = {}) :
    # name = "SocialMediaTextAnalyzer"
    # tags = {"nlp.framework": "Spark NLP"}
    # desc = "This sentiment analysis model classifies the tone-happy, sad, angry."
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())
    client.create_registered_model(name, tags, desc)
    

def create_registered_model_if_not_exists(login_id, login_pwd, name, desc : str = '', tags : dict = {}) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)
    os.environ['MLFLOW_TRACKING_USERNAME'] = login_id
    os.environ['MLFLOW_TRACKING_PASSWORD'] = login_pwd

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())

    try:
        model = client.get_registered_model(name)
        return model.name
    except MlflowException :
        pass

    client.create_registered_model(name, tags, desc)
    model = client.get_registered_model(name)
    return model.name

def get_registered_model(login_id, login_pwd, name) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())
    try:
        model = client.get_registered_model(name)
        return model.name
    except MlflowException :
        return None

def update_artifact_location(experiment_id, artifact_location) :
    
    #establishing the connection
    conn = psycopg2.connect(
       database="mlflow", user='mlflow', password='mlflow', host='apache-airflow-postgresql.default.svc.cluster.local', port= '5432'
    )
    
    #Setting auto commit false
    conn.autocommit = True

    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    artifact_location = os.path.join(artifact_location, str(experiment_id))

    update_sql = f"""
    UPDATE experiments
    SET artifact_location = '{artifact_location}'
    WHERE experiment_id = {experiment_id}
    """

    #Doping EMPLOYEE table if already exists.
    cursor.execute(update_sql)

    #Commit your changes in the database
    conn.commit()
    
    #Closing the connection
    conn.close()