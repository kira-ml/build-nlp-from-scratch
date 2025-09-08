import os
from pathlib import Path




PROJECT_ROOT = Path(r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics")



DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"



RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics\data\processed\emails_sampled_5k.csv")

THREAD_STORE_PATH = os.path.join(PROCESSED_DATA_DIR, "canonical_email_threads.csv")




BODY_COLUMN = 'body'



SUBJECT_COLUMN = 'subject'

MESSAGE_ID_COLUMN = 'Message-ID'

IN_REPLY_TO_COLUMN = 'In-Reply-To'


DATE_COLUMN = 'date'



if not os.path.isdir(RAW_DATA_PATH):
    print(f"Warning: the configured raw data path does not exist: {RAW_DATA_DIR}")

    print("Please check the path in config.py")
