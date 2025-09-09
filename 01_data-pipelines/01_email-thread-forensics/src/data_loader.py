import pandas as pd
import csv
import config
import sys




csv.field_size_limit(10_000_000)  # Set to 10 million characters, safe for large email fields
def load_email_data(file_path):


    dataframe = pd.read_csv(file_path, engine='python')


    return dataframe



def inspect_data(dataframe):


    print("INITIAL DATA INSPECTION")


    print(f"Dataset shape: {dataframe.shape}")

    print(f" This dataset has {dataframe.shape[0]} rows (emails) and {dataframe.shape[1]} column (features)")



    print("Columns and datatypes: ")

    print(dataframe.dtypes)

    print("First 5 rows:")
    print(dataframe.head())
    print("Inspection complete")



if __name__ == "__main__":

    print("Testing the data loader Module")

    data_file_path = config.RAW_DATA_PATH

    print("Attempting to load the data from: {data_file_path}")

    print("Loading data...")

    dataframe = load_email_data(data_file_path)

    print("Data successfully loaded")


    print("Inspecting the data")

    print(inspect_data(dataframe))