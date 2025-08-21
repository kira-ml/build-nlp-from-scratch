import pandas as pd
from email import policy
from email.parser import BytesParser


def load_and_validate_emails(file_path: str) -> pd.DataFrame:

    df = pd.read_csv(file_path)

    required_column = 'raw_email'
    if required_column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Missing required columns '{required_column}'. "
            f"Available Columns: {available}"
        )
    
    print(f"Loaded {len(df)} emails. Columns: {list(df.columns)}")
    return df


def parse_single_email(raw_email: str) -> dict:

    email_bytes =  raw_email.encode('utf-8', errors='replace')


    msg = BytesParser(policy=policy.default).parsebytes(email_bytes)


    headers = {k: str(v) for k, v in msg.items()}


    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                break
    
    else:
        body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
    
    return {
        "headers": headers,
        "body": body[:500] + "..." if len(body) > 500 else body

    }


if __name__ == "__main__":
    file_path = r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics\data\processed\emails_sampled_5k.csv"
    df = pd.read_csv(file_path)
    email_df = load_and_validate_emails(file_path)

    result = parse_single_email(df['raw_email'].iloc[0])


    print("\nParsed Headers:")
    for key, value in list(result["headers"].items())[:5]:
        print(f" {key}: {value[:50]}{'...' if len(value) > 50 else ''}")

    print("\nBody preview:")
    print(result["body"])


    print("\nFirst email snippet:")
    print(email_df['raw_email'].iloc[0][:200] + "...")


