import os
import pandas as pd


DATASET_PATH = r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics\data\raw\emails.csv"



if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

else:
    print(f"Dataset found {DATASET_PATH}")




preview_df = pd.read_csv(DATASET_PATH, nrows=5)

print(f"Dataset preview loaded")
print(preview_df.head())
print(f"Columns: {list(preview_df.columns)}")


emails_df = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)


print("Full dataset loaded")
print(f"Total rows: {len(emails_df)}")



def sample_dataset(
        dataframe: pd.DataFrame,
        sample_size: int,
        random_seed: int = 42

) -> pd.DataFrame:


    if sample_size > len(dataframe):
        raise ValueError("Sample size cannot be exceed in dataset size")
    
    sampled_df = dataframe.sample(n=sample_size, random_state=random_seed)


    return sampled_df

sampled_emails_df = sample_dataset(emails_df, sample_size=5000)
print(f"Sampled dataset created")
print(sampled_emails_df.head())



OUTPUT_DIR = r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics\data\processed"


os.makedirs(OUTPUT_DIR, exist_ok=True)


SAMPLED_PATH = os.path.join(OUTPUT_DIR, "emails_sampled_5k.csv")
sampled_emails_df.to_csv(SAMPLED_PATH, index=False)

print(f"Sampled dataset saved at: {SAMPLED_PATH}")


reloaded_sample_df = pd.read_csv(SAMPLED_PATH)


assert len(reloaded_sample_df) == 5000, "Sample size mismatch!"
assert len(reloaded_sample_df.columns) == len(emails_df.columns), "Number of columns mismatch"
assert list(reloaded_sample_df.columns) == list(emails_df.columns), "Column names mismatch"

print("Sanity check passed: sampled dataset is consistent and valid")
