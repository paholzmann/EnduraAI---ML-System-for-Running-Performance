import pandas as pd
import numpy as np

class UTMBFeatures:
    def __init__(self):
        pass

    def calculate_race_effort(self, utmb_df: pd.DataFrame) -> pd.DataFrame:
        """
        1km = 1km effort
        100m+ = 1km effort
        1km with 100m+ = 2km effort
        Example:
            100km with 6000m+ = 160km effort
        """
        utmb_df = utmb_df.copy()
        utmb_df["Race_Effort"] = utmb_df["Distance"] + (utmb_df["Elevation_Gain"] / 100)
        return utmb_df
    
    def calculate_race_results_features(self, utmb_df: pd.DataFrame) -> pd.DataFrame:
        utmb_df = utmb_df.copy()
        utmb_df["Results"] = utmb_df["Results"].apply(lambda x: [t for t in x if pd.notna(t) and t > 0] if isinstance(x, list) else [])
        utmb_df["Winning_time"] = utmb_df["Results"].apply(lambda x: min(x) if x else np.nan)
        utmb_df["Median_time"] = utmb_df["Results"].apply(lambda x: np.median(x) if x else np.nan)
        utmb_df["Slowest_time"] = utmb_df["Results"].apply(lambda x: max(x) if x else np.nan)
        utmb_df["Winning_Slowest_Time_Range"] = utmb_df["Slowest_time"] - utmb_df["Winning_time"]
        utmb_df["Top_10_Percent_Average_Time"] = utmb_df["Results"].apply(lambda x: ([t for t in x if t <= np.percentile(x, 10)]) if x else np.nan)
        utmb_df["Podium_Average_Time"] = utmb_df["Results"].apply(lambda x: np.mean(sorted(x)[:3]) if len(x) >= 3 else np.nan)
        utmb_df["Podium_vs_Field"] = (utmb_df["Podium_Average_Time"] / utmb_df["Median_time"])
        return utmb_df