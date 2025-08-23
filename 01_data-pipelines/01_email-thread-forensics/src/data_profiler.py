import pandas as pd
import json

def load_email_dataset(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {df.shape[0]} emails and {df.shape[1]} fields.")
        print(f"first 5 rows")
        print(df.head())
        print("\nData types and missing per column:")
        df.info()
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

dataset_path = r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics\data\processed\emails_sampled_5k.csv"
emails_df = load_email_dataset(dataset_path)

def analyze_field_completeness(df: pd.DataFrame) -> pd.DataFrame:
    completeness = df.notnull().mean() * 100
    completeness_df = completeness.reset_index()
    completeness_df.columns = ["field_name", "completeness_percentage"]
    print("Field name completeness:\n", completeness_df)
    return completeness_df

completeness_report = analyze_field_completeness(emails_df)

def detect_encoding_issues(df: pd.DataFrame, text_column: str = "body") -> pd.DataFrame:
    encoding_issues = []
    
    # Fix: iteritems() is deprecated, use items() instead
    for idx, text in df[text_column].items():
        try:
            if pd.isna(text):
                continue
            text.encode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            encoding_issues.append(idx)
    
    issues_df = df.loc[encoding_issues]
    print(f"Found {len(issues_df)} emails with encoding issues")
    return issues_df  # Fix: Return the DataFrame

# Fix: Column name should be "message" not "body"
encoding_issues_df = detect_encoding_issues(emails_df, text_column="message")

def identify_duplicate_emails(df: pd.DataFrame, subset_fields: list = ["file", "message"]) -> pd.DataFrame:
    # Fix: Use existing columns "file" and "message" instead of "subject" and "body"
    duplicates = df[df.duplicated(subset=subset_fields, keep=False)]
    print(f"Found {len(duplicates)} duplicate emails based on {subset_fields}")
    return duplicates  # Fix: Return the DataFrame

duplicate_emails_df = identify_duplicate_emails(emails_df)

def generate_quality_report(
        df: pd.DataFrame, completeness_df: pd.DataFrame,
        encoding_issues_df: pd.DataFrame, duplicate_df: pd.DataFrame,
        report_path_json: str = "data_quality_report.json",
        metrics_path_jsonl: str = "quality_metrics.jsonl"
):
    # Fix: Use the correct column name from completeness_df
    completeness_dict = completeness_df.set_index("field_name")["completeness_percentage"].to_dict()
    
    report_summary = {
        "total_emails": df.shape[0],
        "field_completeness": completeness_dict,
        "num_encoding_issues": len(encoding_issues_df),
        "num_duplicates": len(duplicate_df),
        "recommendations": [
            "Review fields with completeness < 90%",
            "Fix encoding issues before NLP processing",
            "Remove duplicates for unbiased statistics"
        ]
    }

    # Fix: Corrected variable names and logic
    with open(report_path_json, "w", encoding="utf-8") as f:
        json.dump(report_summary, f, indent=4)
    print(f"Saved data quality summary to {report_path_json}")

    with open(metrics_path_jsonl, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            email_score = {
                "email_index": idx,
                "missing_field": int(row.isnull().sum()),
                "has_encoding_issues": idx in encoding_issues_df.index.tolist(),
                "is_duplicate": idx in duplicate_df.index.tolist()
            }
            f.write(json.dumps(email_score) + "\n")
        print(f"Saved per-email quality metrics to {metrics_path_jsonl}")

# Fix: Pass the correct variable names
generate_quality_report(emails_df, completeness_report, encoding_issues_df, duplicate_emails_df)