from kidney_classifier.components.prepare_base_model import PrepareBaseModel
from kidney_classifier.configuration.configuration import ConfigurationManager


class PrepareBaseModelTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

