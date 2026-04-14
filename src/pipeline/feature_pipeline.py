import pandas as pd
import logging
from src.logging.logger import Logger
from src.feature_engineering.utmb_features import UTMBFeatures

class UTMBFeaturePipeline:
    def __init__(self):
        self.logger = Logger(name="Feature pipeline", level=logging.DEBUG).logger
        self.utmb_features = UTMBFeatures()

    def run_feature_pipeline(self, utmb_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(f"Running UTMB feature pipeline")
        utmb_df = self.utmb_features.calculate_race_effort(utmb_df=utmb_df)
        utmb_df = self.utmb_features.calculate_race_result_features(utmb_df=utmb_df)
        utmb_df = self.utmb_features.calculate_normalized_features(utmb_df=utmb_df)
        utmb_df = self.utmb_features.calculate_competition_features(utmb_df=utmb_df)
        utmb_df = self.utmb_features.calculate_difficulty_features(utmb_df=utmb_df)
        return utmb_df