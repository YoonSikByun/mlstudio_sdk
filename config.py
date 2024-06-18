import jsons
from mlstudio_sdk.common import SingletonType

config_path = '/opt/mlstudio/config/mlstudio-config.json'

class Config(object, metaclass=SingletonType) :
    def __init__(self):
        self.config = None
        self.load()

    def load(self) :
        with open(config_path, 'r') as f :
            self.config = jsons.loads(f.read())

    def get_mlflow_tracking_uri(self) :
        ip = self.config['db']['cluster']['ip']
        port = self.config['db']['cluster']['port']
        user_id = self.config['db']['user']['mlflow']['id']
        user_pwd = self.config['db']['user']['mlflow']['password']
        dbname = self.config['db']['database']['mlflow']['tracking']

        return f"postgresql://{user_id}:{user_pwd}@{ip}:{port}/{dbname}"

    def get_mlflow_artifact_url(self) :
        return self.config['mlflow']['artifact_root']

if __name__ == '__main__':
    config = Config()
    print(config.config)
    print(config.get_mlflow_tracking_uri())
    print(config.get_mlflow_artifact_url())
