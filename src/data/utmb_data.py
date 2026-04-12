import pandas as pd
import ast
import re
from .file_handler import FileHandler

class UTMBData:
    def __init__(self):
        pass

    def clean_raw_df(self, utmb_df: pd.DataFrame, columns_to_expand: list) -> pd.DataFrame:
        utmb_df = utmb_df.T
        utmb_df = utmb_df.rename(columns={"City / Country": "Race_Country", "Elevation Gain": "Elevation_Gain", "N Results": "N_Results", "Race Category": "Race_Category", "Race Title": "Race_Title"})
        for col in columns_to_expand:
            expanded_df = utmb_df[col].apply(pd.Series)
            expanded_df.columns = [f"{col}_{str(subcol).strip().replace(' ', '_')}" for subcol in expanded_df.columns]
            utmb_df = pd.concat([utmb_df, expanded_df], axis=1)
            utmb_df = utmb_df.drop(columns=[col])
        return utmb_df
    
    def load_processed_df(self, filepath: str = "training_data/utmb/processed/utmb-race-data-processed.csv") -> pd.DataFrame:
        df = pd.read_csv(filepath)
        return df    

class CleanUTMBData:
    def __init__(self):
        pass

    def remove_str_from_numeric_col(self, utmb_df: pd.DataFrame, columns: list = ["Distance", "Elevation_Gain"]) -> pd.DataFrame:
        def extract_number(val):
            if pd.isna(val):
                return None
            match = re.findall(r"\d+\.?\d*", str(val))
            return float(match[0]) if match else None
        for col in columns:
            utmb_df[col] = utmb_df[col].apply(extract_number)
        return utmb_df

    def replace_nulls_by_prefix(self, utmb_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        for col in utmb_df.columns:
            if prefix.lower() in col.lower():
                utmb_df[col] = utmb_df[col].fillna(0)
        return utmb_df
    
    def parse_race_results(self, utmb_df: pd.DataFrame) -> pd.DataFrame:
        utmb_df["Results"] = utmb_df["Results"].apply(ast.literal_eval)
        return utmb_df


# file_handler = FileHandler()
# utmb_df = file_handler.read_json_as_df(json_filepath="training_data/utmb/raw/utmb-race-data-raw.json")
# # print(utmb_df)
# utmb_df = UTMBData().clean_df(utmb_df=utmb_df, columns_to_expand=["Age", "Country", "Sex"])
# print(utmb_df)
# print(utmb_df.columns)