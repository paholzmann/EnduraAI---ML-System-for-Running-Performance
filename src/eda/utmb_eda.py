import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from ..data.file_handler import FileHandler
from ..data.utmb_data import UTMBData, CleanUTMBData
from ..feature_engineering.utmb_features import UTMBFeatures
from ..logging.logger import Logger

class UTMB_EDA:
    def __init__(self):
        self.logger = Logger(name="Utmb EDA", level=logging.DEBUG).logger

    def plot_race_category(self, utmb_df: pd.DataFrame) -> None:
        self.logger.debug(f"Plotting race categories (countplot)")
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10,5)
        order = utmb_df["Race_Category"].value_counts().index
        sns.countplot(data=utmb_df, x="Race_Category", order=order)
        plt.title("Number of races per race category")
        plt.ylabel("Race category")
        plt.xlabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    def plot_race_distance(self, utmb_df: pd.DataFrame) -> None:
        self.logger.debug(f"Plotting race distances (histplot)")
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 5)
        sns.histplot(data=utmb_df, x="N_Results", bins=20, kde=True)
        plt.title("Distribution of race distance")
        plt.xlabel("Distance in km")
        plt.ylabel("Number of races")
        plt.show()

    def plot_distance_vs_elevation_gain(self, utmb_df: pd.DataFrame) -> None:
        self.logger.debug(f"Plotting race distances vs. elevation gain (scatterplot)")
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 5)
        sns.scatterplot(data=utmb_df, x="Distance", y="Elevation_Gain")
        plt.plot([utmb_df["Distance"].min(), utmb_df["Distance"].max()], [utmb_df["Distance"].min(), utmb_df["Distance"].max()], linestyle="--")
        plt.title("Distance vs. elevation gain")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Distance in km")
        plt.ylabel("Elevation gain in m")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_boxplot(self, utmb_df: pd.DataFrame, x: str, x_label: str, title: str) -> None:
        self.logger.debug(f"Plotting {x} with outliers (boxplot)")
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 5)
        sns.boxplot(data=utmb_df, x=x)
        plt.xlabel(x_label)
        plt.title(title)
        plt.show()

    def plot_average_race_time_by_category(self, utmb_df: pd.DataFrame) -> pd.DataFrame:
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 5)
        sns.boxplot(data=utmb_df, x="Race_Category", y="Median_Time")
        plt.xlabel("Race category")
        plt.ylabel("Average finishing time in h")
        plt.yscale("log")
        plt.title("Race category vs average finishing time in h")
        plt.show()

file_handler = FileHandler()
utmb_data = UTMBData()
clean_utmb_data = CleanUTMBData()
utmb_features = UTMBFeatures()
utmb_df = utmb_data.load_processed_df()
utmb_df = clean_utmb_data.remove_str_from_numeric_col(utmb_df=utmb_df)
utmb_df = utmb_features.calculate_race_effort(utmb_df=utmb_df)
utmb_df = clean_utmb_data.parse_race_results(utmb_df=utmb_df)
utmb_df = utmb_features.calculate_race_result_features(utmb_df=utmb_df)
utmb_df = utmb_features.calculate_normalized_features(utmb_df=utmb_df)
utmb_df = utmb_features.calculate_competition_features(utmb_df=utmb_df)
utmb_df = clean_utmb_data.replace_nulls_by_prefix(utmb_df=utmb_df, prefix="country")
utmb_df = utmb_features.calculate_difficulty_features(utmb_df=utmb_df)
print(utmb_df)
utmb_eda = UTMB_EDA()
utmb_eda.plot_race_category(utmb_df=utmb_df)
utmb_eda.plot_boxplot(utmb_df=utmb_df, x="Distance", x_label="Distance", title="Distance-Distribution with outliers")
utmb_eda.plot_boxplot(utmb_df=utmb_df, x="Elevation_Gain", x_label="Elevation gain", title="Elevation gain distribution with outliers")
utmb_eda.plot_race_distance(utmb_df=utmb_df)
utmb_eda.plot_distance_vs_elevation_gain(utmb_df=utmb_df)
utmb_eda.plot_average_race_time_by_category(utmb_df=utmb_df)