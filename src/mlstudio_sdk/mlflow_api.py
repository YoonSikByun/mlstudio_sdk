import os
import mlflow
from mlflow import MlflowClient
from mlflow.server import get_app_client
import psycopg2
from mlstudio_sdk.config import Config
from mlflow.exceptions import MlflowException
from mlflow.server.auth.client import AuthServiceClient

config = Config()

# Permission      |  Can read | Can update | Can delete | Can manage
# READ               Yes          No           No            No
# EDIT               Yes          Yes          No            No
# MANAGE             Yes          Yes          Yes           Yes
# NO_PERMISSIONS     No           No           No            No  
read_permission = ['READ', 'EDIT', 'MANAGE']
update_permission = ['EDIT', 'MANAGE']
delete_permission = ['MANAGE']
manage_permission = ['MANAGE']

def set_tracking_user_env(login_id='', login_pwd='') :
    if login_id:
        os.environ['MLFLOW_TRACKING_USERNAME'] = login_id
    if login_pwd:
        os.environ['MLFLOW_TRACKING_PASSWORD'] = login_pwd

########################################
# 사용자 관리
########################################
# ML Flow 사용자를 추가한다.
def create_user(login_id, login_pwd, user_id, user_pwd) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    auth_client = get_app_client("basic-auth", tracking_uri=config.get_mlflow_server_url())

    auth_client.create_user(username=user_id, password=user_pwd)

# ML Flow 사용자를 삭제한다.
def delete_user(login_id, login_pwd, user_id) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    auth_client = get_app_client("basic-auth", tracking_uri=config.get_mlflow_server_url())
    auth_client.delete_user(username=user_id)

# 사용자 관리자 승격/취소
def update_user_admin(login_id, login_pwd, user_id, is_admin) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    auth_client = get_app_client("basic-auth", tracking_uri=config.get_mlflow_server_url())
    auth_client.update_user_admin(username=user_id, is_admin=is_admin)


########################################
# experiment 생성/삭제
########################################
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

########################################
# registered model 생성/삭제
########################################
def create_registered_model(login_id, login_pwd, name, desc : str = '', tags : dict = {}) :
    # name = "SocialMediaTextAnalyzer"
    # tags = {"nlp.framework": "Spark NLP"}
    # desc = "This sentiment analysis model classifies the tone-happy, sad, angry."
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())
    client.create_registered_model(name, tags, desc)

def create_registered_model_if_not_exists(login_id, login_pwd, name, desc : str = '', tags : dict = {}) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

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

def delete_registered_model(login_id, login_pwd, registered_model_name) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())
    try:
        model = client.get_registered_model(registered_model_name)
    except MlflowException :
        return

    client.delete_registered_model(registered_model_name)

########################################
# 접근권한 관리
########################################
# registered model 접근권한 설정
def apply_registered_model_permission(login_id, login_pwd, registered_model_name, user_id, perssion) :
    # Permission      |  Can read | Can update | Can delete | Can manage
    # READ               Yes          No           No            No
    # EDIT               Yes          Yes          No            No
    # MANAGE             Yes          Yes          Yes           Yes
    # NO_PERMISSIONS     No           No           No            No  
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)
    auth_client = AuthServiceClient(tracking_uri=config.get_mlflow_server_url())

    return auth_client.create_registered_model_permission(name=registered_model_name, username=user_id, permission=perssion)

# registered model 접근권한 설정 변경
def update_registered_model_permission(login_id, login_pwd, registered_model_name, user_id, perssion) :
    # Permission      |  Can read | Can update | Can delete | Can manage
    # READ               Yes          No           No            No
    # EDIT               Yes          Yes          No            No
    # MANAGE             Yes          Yes          Yes           Yes
    # NO_PERMISSIONS     No           No           No            No  
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)
    auth_client = AuthServiceClient(tracking_uri=config.get_mlflow_server_url())

    auth_client.update_registered_model_permission(name=registered_model_name, username=user_id, permission=perssion)

# registered model 접근권한 취소
def cancel_registered_model_permission(login_id, login_pwd, registered_model_name, user_id) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())
    auth_client = AuthServiceClient(tracking_uri=config.get_mlflow_server_url())

    try:
        model = client.get_registered_model(registered_model_name)
    except MlflowException :
        return

    auth_client.delete_registered_model_permission(name=registered_model_name, username=user_id)

# experiment 접근권한 설정
def apply_experiment_permission(login_id, login_pwd, experiment_name, user_id, permission) :
    # Permission      |  Can read | Can update | Can delete | Can manage
    # READ               Yes          No           No            No
    # EDIT               Yes          Yes          No            No
    # MANAGE             Yes          Yes          Yes           Yes
    # NO_PERMISSIONS     No           No           No            No
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_server_url())
    auth_client = AuthServiceClient(tracking_uri=config.get_mlflow_server_url())

    experiment_details = client.get_experiment_by_name(experiment_name)

    if experiment_details :
        experiment_id = experiment_details.experiment_id
    else :
        raise Exception(f'{experiment_name} does not exist.')

    return auth_client.create_experiment_permission(experiment_id=experiment_id, username=user_id, permission=permission)

# experiment 접근권한 설정 변경
def update_experiment_permission(login_id, login_pwd, experiment_name, user_id, permission) :
    # Permission      |  Can read | Can update | Can delete | Can manage
    # READ               Yes          No           No            No
    # EDIT               Yes          Yes          No            No
    # MANAGE             Yes          Yes          Yes           Yes
    # NO_PERMISSIONS     No           No           No            No
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_server_url())
    auth_client = AuthServiceClient(tracking_uri=config.get_mlflow_server_url())

    experiment_details = client.get_experiment_by_name(experiment_name)

    if experiment_details :
        experiment_id = experiment_details.experiment_id
    else :
        raise Exception(f'{experiment_name} does not exist.')

    auth_client.update_experiment_permission(experiment_id=experiment_id, username=user_id, permission=permission)


# experiment 접근권한 취소
def cancel_experiment_permission(login_id, login_pwd, experiment_name, user_id) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())
    auth_client = AuthServiceClient(tracking_uri=config.get_mlflow_server_url())

    experiment_details = client.get_experiment_by_name(experiment_name)

    if experiment_details :
        experiment_id = experiment_details.experiment_id
    else :
        return

    auth_client.delete_experiment_permission(experiment_id=experiment_id, username=user_id)

def get_all_experiments(login_id, login_pwd) :
    set_tracking_user_env(login_id=login_id, login_pwd=login_pwd)

    client = MlflowClient(tracking_uri=config.get_mlflow_tracking_uri())
    return client.search_experiments()

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

if __name__ == "__main__":
    login_id='tes1'
    login_pwd='test1'

    user_id='test2'
    user_pwd='test2'

    experiment_name='MLflow Quickstart2'
    registered_model_name='test3-regitered-models'

    r = get_all_experiments(login_id, login_pwd)
    for i in r:
        print(i.experiment_id)
    # r = delete_experiment(login_id, login_pwd, experiment_name='실험명 한글 사용이 되는건가?')
    # r = delete_registered_model(login_id, login_pwd, '모델 레지스트리 한글 사용이 되는건가?')
    # r = create_user(login_id, login_pwd, user_id, user_pwd)
    # r = delete_user(login_id, login_pwd, user_id)
    # r = update_user_admin(login_id, login_pwd, user_id, is_admin=False)
    # r = apply_registered_model_permission(login_id, login_pwd, registered_model_name, user_id=user_id, perssion='MANAGE')
    # r = update_registered_model_permission(login_id, login_pwd, registered_model_name, user_id=user_id, perssion='MANAGE')
    # r = cancel_registered_model_permission(login_id, login_pwd, registered_model_name, user_id)
    # r = apply_experiment_permission(login_id=login_id, login_pwd=login_pwd, experiment_name=experiment_name, user_id=user_id, permission='MANAGE')
    # r = update_experiment_permission(login_id=login_id, login_pwd=login_pwd, experiment_name=experiment_name, user_id=user_id, permission='MANAGE')
    # r = cancel_experiment_permission(login_id, login_pwd, experiment_name, user_id)

    print(r)
