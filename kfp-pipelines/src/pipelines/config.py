# Pipeline Config
PROJECT_ID = 'glossy-alliance-423610-i9'
REGION = 'us-central1'
STAGING_BUCKET = 'gs://minhas-pipelines'
ARTIFACTS_BUCKET = 'kfp-artifacts'
LOCAL_PIPELINE_FILE = 'outputs/my-scikit-learn-model.yaml'

KFP_PYTHON_IMAGE = 'python:3.10'
PIPE_ARTIFACT_REPO = 'kfp-pipelines'
PIPE_ARTIFACT_REPO_REGION = 'us-central1'
PIPELINE_REGISTRY_URL = f'https://{PIPE_ARTIFACT_REPO_REGION}-kfp.pkg.dev/{PROJECT_ID}/{PIPE_ARTIFACT_REPO}'
PRED_CONTAINER_URL = 'us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest'


RANDOM_STATE = 42
PARAMS = dict(
    target_col='target'
)
DATA_COMPONENT_PACKAGE_REQUIREMENTS = ['scikit-learn']
TRAIN_COMPONENT_PACKAGE_REQUIREMENTS = ['scikit-learn', 'joblib', 'google-cloud-platform', 'pandas']

VERTEX_MODEL_NAME = 'my-scikit-learn-pipeline'
EXPERIMENT_NAME = f'{VERTEX_MODEL_NAME}-exp'
EXPERIMENT_MODEL_TYPE = 'logiticregression'
EXPERIMENT_FRAMEWORK = 'scikit-learn'