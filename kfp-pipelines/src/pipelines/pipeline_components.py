from typing import List

from kfp.v2 import dsl as dsl
from kfp.v2.dsl import Output, Dataset, Input, OutputPath, InputPath

from src.pipelines import config


@dsl.component(
    base_image=config.KFP_PYTHON_IMAGE,
    packages_to_install=config.DATA_COMPONENT_PACKAGE_REQUIREMENTS
)
def data_component(
        base_output: Output[Dataset]
):

    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame

    df.to_csv(base_output.path, index=False)


@dsl.component(
    base_image=config.KFP_PYTHON_IMAGE,
    packages_to_install=config.TRAIN_COMPONENT_PACKAGE_REQUIREMENTS
)
def train_model(
        input_dataset: Input[Dataset],
        fit_model_output_path: OutputPath("Model"),
        target_col: List[str],
        random_state: int,
        project_id: str,
        experiment_name: str,
        experiment_model_type: str,
        experiment_framework: str,
        vertex_region: str,
        vertex_model_name: str,
        staging_bucket: str
):
    import os
    import joblib
    from datetime import datetime
    from google.cloud import aiplatform
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn import __version__ as sklearnversion

    train_dataset = pd.read_csv(input_dataset.path)

    features = train_dataset.drop(columns=target_col).copy()
    target = train_dataset[target_col]

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        random_state=random_state
    )

    # Vertex AI Config

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    aiplatform.init(
        project=project_id,
        location=vertex_region,
        experiment=f'{experiment_name}-{target_col}',
        staging_bucket=f'{staging_bucket}/staging'
    )

    with aiplatform.start_run(run=f'{experiment_name}-{target_col}-{timestamp}') as exp_run:
        experiment_params = {
            'model_type': experiment_model_type,
            'framework': experiment_framework
        }

        aiplatform.log_params(experiment_params)

        # Model training
        clf_params = dict(random_state=random_state)
        clf = LogisticRegression(**clf_params)

        aiplatform.log_params(**clf_params)
        aiplatform.log_params({'sklearn_version': sklearnversion})

        clf.fit(features_train, target_train)

        os.makedirs(fit_model_output_path)
        joblib.dump(clf, f'{fit_model_output_path}/model.pkl')
        print(f"Saving model to {fit_model_output_path}")

        # TODO: metrics

        aiplatform.log_model(
            model=clf,
            display_name=f'{vertex_model_name}-{target_col}'
        )


@dsl.component(
    base_image=config.KFP_PYTHON_IMAGE,
    packages_to_install=config.DATA_COMPONENT_PACKAGE_REQUIREMENTS
)
def upload_model(
        model_url: InputPath('Model'),
        name: str,
        description: str,
        project_id: str,
        region: str,
        prediction_container_url: str
):
    from google.cloud import aiplatform
    registered_models = aiplatform.Model.list(
        filter=f"display_name={name}"
    )
    parent_model = registered_models[0].resource_name if registered_models else None
    gs_uri = str(model_url).replace('/gcs/', 'gs://')
    aiplatform.Model.upload(
        project=project_id,
        location=region,
        display_name=name,
        parent_model=parent_model,
        serving_container_image_uri=prediction_container_url,
        artifact_uri=gs_uri,
        description=description
    )

