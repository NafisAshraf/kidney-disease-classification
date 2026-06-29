import os
from pathlib import Path

from kidney_classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from kidney_classifier.entity.config_entity import (
    DataIngestionConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
)
from kidney_classifier.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        training_data = os.path.join(
            self.config.data_ingestion.unzip_dir,
            "kidney-ct-scan-image",
        )
        create_directories([Path(training.root_dir)])
        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        return EvaluationConfig(
            path_of_model=Path(config.path_of_model),
            training_data=Path(config.training_data),
            mlflow_uri=config.mlflow_uri,
            all_params=dict(self.params),
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )

