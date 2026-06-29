from kidney_classifier.components.model_training import Training
from kidney_classifier.configuration.configuration import ConfigurationManager


class ModelTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

