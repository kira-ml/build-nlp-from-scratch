import os
from pathlib import Path




PROJECT_ROOT = Path(r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics")



DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_PATH = r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics\data\raw\emails_sampled_5k.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"





THREAD_STORE_PATH = os.path.join(PROCESSED_DATA_DIR, "canonical_email_threads.csv")




BODY_COLUMN = 'message'  # Updated to match actual CSV column name



SUBJECT_COLUMN = 'subject'

MESSAGE_ID_COLUMN = 'Message-ID'

IN_REPLY_TO_COLUMN = 'In-Reply-To'


DATE_COLUMN = 'date'



if not os.path.isfile(RAW_DATA_PATH):
    print(f"Warning: the configured raw data path does not exist: {RAW_DATA_PATH}")

    print("Please check the path in config.py")
