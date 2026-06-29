from kidney_classifier.components.model_evaluation_mlflow import Evaluation
from kidney_classifier.configuration.configuration import ConfigurationManager


class EvaluationPipeline:
    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()

