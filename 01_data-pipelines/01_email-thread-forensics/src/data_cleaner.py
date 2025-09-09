import config
import data_loader


import pandas as pd

import re



def clean_email_data(dataframe):


    cleaned_df = dataframe.copy()

    print("Starting data cleaning process...")

    print(f"Initial shape:", cleaned_df.shape)


    print("\n Cleaning text data...")


    cleaned_df[config.BODY_COLUMN] = cleaned_df[config.BODY_COLUMN].astype(str)




    cleaned_df[config.BODY_COLUMN] = cleaned_df[config.BODY_COLUMN].str.lower()


    print(" -Converted to lowercase")
    print("Handling missing values...")


    initial_missing = cleaned_df[config.BODY_COLUMN].isna().sum()


    print(f" -Found {initial_missing} emails with missing Body text")



    cleaned_df = cleaned_df.dropna(subset=[config.BODY_COLUMN]).copy()


    empty_mask = cleaned_df[config.BODY_COLUMN].str.strip() == ""
     
    cleaned_df = cleaned_df[~empty_mask].copy()


    print(f" -Removed {initial_missing + empty_mask.sum()} invalid emails")
    print(f" -New shape {cleaned_df.shape}")



    cleaned_df[config.BODY_COLUMN] = cleaned_df[config.BODY_COLUMN].apply(
        lambda text: text.encode('ascii', 'ignore').decode('ascii')


    )

    print(" -Removed non-ASCII characters")

    return cleaned_df