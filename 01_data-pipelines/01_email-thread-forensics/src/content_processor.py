import pandas as pd
import os
import re
import email.utils
import hashlib
from datetime import datetime
from bs4 import BeautifulSoup

file_path = r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics\data\processed\emails_sampled_5k.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at: {file_path}")

raw_emails_df = pd.read_csv(file_path, nrows=10)

print("Columns in dataset:", raw_emails_df.columns.tolist())

essential_columns = ['message_id', 'from', 'to', 'date', 'subject', 'body']
emails_df = raw_emails_df[essential_columns].copy()

print(f"Loaded {len(emails_df)} sample emails from dataset")

def extract_email_body(html_body):
    if not isinstance(html_body, str):
        return ""
    
    soup = BeautifulSoup(html_body, 'html.parser')
    
    # Replace <br> tags with newlines
    for br in soup.find_all('br'):
        br.replace_with('\n')
    
    # Replace <p> tags with double newlines for paragraph separation
    for p in soup.find_all('p'):
        p.insert_after('\n\n')
    
    # Get text and clean up extra whitespace
    text = soup.get_text()
    lines = [line.strip() for line in text.splitlines()]
    cleaned = [line for line in lines if line]  # Remove empty lines
    return '\n'.join(cleaned)

sample_body = emails_df.iloc[0]['body']
cleaned_body = extract_email_body(sample_body)

print("Extracted plain text:\n", cleaned_body[:300] + "..." if len(cleaned_body) > 300 else cleaned_body)

def detect_quoted_content(text_body):
    if not isinstance(text_body, str) or not text_body.strip():
        return {'original': '', 'quoted': ''}
    
    quote_lines = []
    original_lines = []
    
    lines = text_body.splitlines()
    in_quote = False
    
    for line in lines:
        # Fixed variable name from 'html_body' to 'text_body'
        if re.match(r'^\s*>', line) or \
           re.search(r'(On\s+[\w\,\:\s]+wrote:|From:\s+.*|Sent:\s+)', line, re.IGNORECASE):
            in_quote = True
        
        if in_quote:
            quote_lines.append(line)
        else:
            original_lines.append(line)
    
    original_text = '\n'.join(original_lines).strip()
    quoted_text = '\n'.join(quote_lines).strip()
    
    return {
        'original': original_text,
        'quoted': quoted_text
    }

# Test the function
quoted_result = detect_quoted_content(cleaned_body)
print("Original content:\n", quoted_result['original'][:200])
print("\nQuoted content:\n", quoted_result['quoted'][:200])



def normalize_headers(raw_from, raw_to, raw_date):

    try:
        from_name, from_addr = email.utils.parseaddr(raw_from)
        cleaned_from = f"{from_name.strip()} <{from_addr.strip()}"

    except:
        cleaned_from = raw_from.strip()

    

    to_list = []
    if isinstance(raw_to, str):
        for addr in raw_to.split(','):
            name, addr = email.utils.parseaddr(addr)
            if addr:
                to_list.append(f"{name.strip()} <{addr.lower()}>")


    cleaned_to = ', '.join(to_list)


    try:
        dt = email.utils.parsedate_to_datetime(raw_date)
        standardized_date = dt.isoformat()
    except:
        standardized_date = None

    return {
        'from': cleaned_from,
        'to': cleaned_to,
        'date': standardized_date
    }


sample_row = emails_df.iloc[0]
normalized = normalize_headers(sample_row['from'], sample_row['to'], sample_row['date'])
print("Normalized headers: \n", normalized)


def generate_content_hash(content_dict):

    concat_string = (
        content_dict['from'].lower().strip() +
        content_dict['to'].lower().strip() +
        content_dict['subject'].strip() +
        content_dict['original_body'].strip() +
        (content_dict['date'] or "")
    )



    hash_obj = hashlib.sha256()
    hash_obj = hashlib.sha256(concat_string.encode('utf-8'))
    return hash_obj.hexdigest()




test_hash_input = {
    'from': normalized['from'],
    'subject': normalized['subject']

}