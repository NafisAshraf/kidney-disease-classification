from kidney_classifier.components.data_ingestion import DataIngestion
from kidney_classifier.configuration.configuration import ConfigurationManager


class DataIngestionTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

