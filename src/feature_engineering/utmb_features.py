import pandas as pd
import numpy as np
import heapq
import logging
from ..logging.logger import Logger

class UTMBFeatures:
    def __init__(self):
        self.logger = Logger(name="Utmb EDA", level=logging.DEBUG).logger

    def calculate_race_effort(self, utmb_df: pd.DataFrame) -> pd.DataFrame:
        """
        1km = 1km effort
        100m+ = 1km effort
        1km with 100m+ = 2km effort
        Example:
            100km with 6000m+ = 160km effort
        """
        self.logger.debug(f"Calculating race effort")
        utmb_df = utmb_df.copy()
        utmb_df["Race_Effort"] = utmb_df["Distance"] + (utmb_df["Elevation_Gain"] / 100)
        return utmb_df

    
    def calculate_race_result_features(self, utmb_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(f"Calculating race result features")
        utmb_df = utmb_df.copy()
        def calculate_features(results):
            if not isinstance(results, list):
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            clean_results = [t for t in results if pd.notna(t) and t > 0]
            if not clean_results:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            winning_time = min(clean_results)
            median_time = np.median(clean_results)
            slowest_time = max(clean_results)
            winning_slowest_time_range = slowest_time - winning_time
            p10 = np.percentile(clean_results, 10)
            top_10_values = [t for t in clean_results if t <= p10]
            top_10_percent_average_time = np.mean(top_10_values) if top_10_values else np.nan
            podium_average_time = np.mean(heapq.nsmallest(3, clean_results)) if len(clean_results) >= 3 else np.nan
            podium_vs_field = (podium_average_time / median_time if pd.notna(podium_average_time) and pd.notna(median_time) and median_time != 0 else np.nan)
            return (
                winning_time,
                median_time,
                slowest_time,
                winning_slowest_time_range,
                top_10_percent_average_time,
                podium_average_time,
                podium_vs_field
            )
        columns_to_create = [
            "Winning_Time",
            "Median_Time",
            "Slowest_Time",
            "Winning_Slowest_Time_Range",
            "Top_10_Percent_Average_Time",
            "Podium_Average_Time",
            "Podium_vs_Field"
        ]
        utmb_df[columns_to_create] = pd.DataFrame(
            utmb_df["Results"].apply(calculate_features).to_list(),
            index=utmb_df.index
        )
        return utmb_df
    
    def calculate_normalized_features(self, utmb_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(f"Calculating normalized features")
        utmb_df = utmb_df.copy()
        utmb_df["Mean_Pace"] = (utmb_df["Median_Time"] / utmb_df["Distance"])
        utmb_df["Winner_Pace"] = (utmb_df["Winning_Time"] / utmb_df["Distance"])
        utmb_df["Effort_Adjusted_Median_Time"] = (utmb_df["Median_Time"] / utmb_df["Race_Effort"])
        utmb_df["Effort_Adjusted_Winning_Time"] = (utmb_df["Winning_Time"] / utmb_df["Race_Effort"])
        return utmb_df
    
    def calculate_competition_features(self, utmb_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(f"Calculating competition features")
        utmb_df["Elite_Gap"] = (utmb_df["Top_10_Percent_Average_Time"] / utmb_df["Winning_Time"])
        utmb_df["Podium_Gap"] = (utmb_df["Podium_Average_Time"] / utmb_df["Winning_Time"])
        utmb_df["Depth_Of_Field"] = (utmb_df["Median_Time"] / utmb_df["Winning_Time"])
        utmb_df["Top_vs_Median"] = (utmb_df["Top_10_Percent_Average_Time"] / utmb_df["Median_Time"])
        return utmb_df