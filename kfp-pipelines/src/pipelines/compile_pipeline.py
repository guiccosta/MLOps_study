from src.pipelines import config
from src.pipelines.pipeline_components import data_component, train_model, upload_model

from kfp.registry import RegistryClient
from kfp.v2 import compiler
from kfp.v2 import dsl


@dsl.pipeline(
    name=config.VERTEX_MODEL_NAME,
    description='my scikit-learn pipeline',
    pipeline_root=f'{config.STAGING_BUCKET}/pipeline-root'
)
def my_scikit_learn_pipeline(
        target_col: str,
        table_path: str,
        project_id: str = config.PROJECT_ID,
        experiment_model_type: str = config.EXPERIMENT_MODEL_TYPE,
        experiment_framework: str = config.EXPERIMENT_FRAMEWORK,
        experiment_name: str = config.EXPERIMENT_NAME,
        vertex_region: str = config.REGION,
        vertex_model_name: str = config.VERTEX_MODEL_NAME,
        staging_bucket: str = config.STAGING_BUCKET,
        random_state: int = config.RANDOM_STATE,
        prediction_container_url: str = config.PRED_CONTAINER_URL

):
    data_step = data_component()

    training_model = train_model(
        input_dataset=data_step.outputs['base_outputs'],
        target_col=target_col,
        random_state=random_state,
        project_id=project_id,
        experiment_model_type=experiment_model_type,
        experiment_framework=experiment_framework,
        experiment_name=experiment_name,
        vertex_region=vertex_region,
        vertex_model_name=vertex_model_name,
        staging_bucket=staging_bucket
    ).set_display_name('my_scikit_learn_model')

    _ = upload_model(
        name=f'{vertex_model_name}-{target_col}',
        model_url=training_model.outputs['fit_model_output_path'],
        description="my scikit learn model",
        project_id=project_id,
        region=vertex_region,
        prediction_container_url=prediction_container_url
    ).set_display_name('upload_my_scikit_learn_model')


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=my_scikit_learn_pipeline,
        pipeline_name=config.VERTEX_MODEL_NAME,
        package_path=config.LOCAL_PIPELINE_FILE,
        pipeline_parameters=config.PARAMS
    )

    client = RegistryClient(
        host=config.PIPELINE_REGISTRY_URL
    )

    templateName, versionName = client.upload_pipeline(
        file_name=config.LOCAL_PIPELINE_FILE,
        tags=['latest']
    )
    print(versionName)
