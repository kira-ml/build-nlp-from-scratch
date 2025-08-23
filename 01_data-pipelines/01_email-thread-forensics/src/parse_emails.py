import os
import pandas as pd
import logging
import email
from email.policy import default
import json
from datetime import datetime
import time


def setup_directories_and_logging():
    """Setup required directories and configure logging"""
    # Define paths
    BASE_DIR = r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics"
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOGS_DIR, f"parse_emails_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Initialized logging and created required directories.")
    return DATA_DIR, LOGS_DIR, timestamp


def load_email_data(data_dir):
    """Load email data from CSV file"""
    input_csv = os.path.join(data_dir, "emails_sampled_5k.csv")
    
    try:
        # Load CSV with explicit handling for bad lines
        df = pd.read_csv(input_csv, on_bad_lines='skip')
        logging.info(f"Successfully loaded {len(df)} emails from {input_csv}")
        
        # Validate required columns (actual columns from sampling.py)
        required_columns = {'file', 'message'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        # Add an ID column based on row index for tracking
        df['id'] = df.index.astype(str)
        
        # Rename columns to match expected names in processing functions
        df = df.rename(columns={
            'file': 'filename', 
            'message': 'raw_text'
        })
        
        logging.info(f"Data validation successful. Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        logging.error(f"Failed to load or validate CSV: {e}")
        raise


def parse_raw_email(raw_text, email_id):
    """
    Parse raw email string into structured components.
    Returns a dict with headers and body.
    """
    try:
        msg = email.message_from_string(raw_text, policy=default)
        
        # Extract all headers
        headers = {}
        for key, value in msg.items():
            headers[key.lower()] = value
            
        # Extract specific headers for easy access
        structured_headers = {
            "from": msg.get("From"),
            "to": msg.get("To"),
            "cc": msg.get("Cc"),
            "bcc": msg.get("Bcc"),
            "date": msg.get("Date"),
            "subject": msg.get("Subject"),
            "message_id": msg.get("Message-ID"),
            "in_reply_to": msg.get("In-Reply-To"),
            "references": msg.get("References"),
        }
        
        # Extract body (both plain text and HTML)
        body_text = ""
        body_html = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                    
                payload = part.get_payload(decode=True)
                if not payload:
                    continue
                    
                try:
                    content = payload.decode('utf-8', errors='replace')
                except Exception:
                    content = str(payload)
                
                if content_type == "text/plain":
                    body_text += content
                elif content_type == "text/html":
                    body_html += content
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                try:
                    content = payload.decode('utf-8', errors='replace')
                except Exception:
                    content = str(payload)
                    
                if msg.get_content_type() == "text/plain":
                    body_text = content
                elif msg.get_content_type() == "text/html":
                    body_html = content
        
        return {
            "id": email_id,
            "headers": structured_headers,
            "all_headers": headers,
            "body_text": body_text.strip(),
            "body_html": body_html.strip(),
            "parsing_success": True,
            "error": None
        }
    
    except Exception as e:
        logging.warning(f"Failed to parse email {email_id}: {str(e)[:100]}...")
        return {
            "id": email_id,
            "headers": {},
            "all_headers": {},
            "body_text": "",
            "body_html": "",
            "parsing_success": False,
            "error": str(e)
        }


def process_all_emails(df):
    """Process all emails and return parsed results"""
    parsed_emails = []
    total = len(df)
    failed_count = 0
    
    logging.info("Starting email parsing process...")
    start_time = time.time()
    
    for idx, row in df.iterrows():
        result = parse_raw_email(row["raw_text"], row["id"])
        
        # Add source filename for traceability
        result["source_filename"] = row["filename"]
        
        parsed_emails.append(result)
        
        if not result["parsing_success"]:
            failed_count += 1

        # Log progress every 500 emails
        if (idx + 1) % 500 == 0 or idx == total - 1:
            logging.info(f"Processed {idx + 1}/{total} emails. Failures so far: {failed_count}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logging.info(f"Parsing complete. {total - failed_count} successful, {failed_count} failed.")
    logging.info(f"Total processing time: {processing_time:.2f} seconds")
    
    return parsed_emails, processing_time


def save_parsed_emails(parsed_emails, data_dir):
    """Save parsed emails to JSONL format"""
    output_jsonl = os.path.join(data_dir, "emails_parsed.jsonl")
    
    try:
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for email_data in parsed_emails:
                f.write(json.dumps(email_data, ensure_ascii=False) + '\n')
        logging.info(f"Parsed emails saved to {output_jsonl}")
    except Exception as e:
        logging.error(f"Failed to write parsed emails: {e}")
        raise


def generate_parsing_report(parsed_emails, processing_time, input_csv, data_dir, timestamp):
    """Generate parsing quality report"""
    from collections import Counter
    
    # Count errors
    error_types = [e["error"] for e in parsed_emails if not e["parsing_success"]]
    error_counter = Counter(error_types[:10])  # Limit to top 10 for readability
    
    success_count = sum(1 for e in parsed_emails if e["parsing_success"])
    total_count = len(parsed_emails)
    success_rate = success_count / total_count if total_count > 0 else 0
    
    parsing_report = {
        "total_emails": total_count,
        "successful_parsing": success_count,
        "failed_parsing": total_count - success_count,
        "success_rate": round(success_rate, 4),
        "error_distribution": dict(error_counter),
        "source_file": input_csv,
        "processing_time_seconds": round(processing_time, 2),
        "processing_timestamp": timestamp
    }
    
    # Save report
    report_path = os.path.join(data_dir, "parsing_report.json")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(parsing_report, f, indent=2)
        logging.info(f"Parsing report saved to {report_path}")
    except Exception as e:
        logging.error(f"Failed to save parsing report: {e}")
        raise


def extract_thread_hints(parsed_emails, data_dir):
    """Extract thread relationship hints from parsed emails"""
    thread_hints_path = os.path.join(data_dir, "thread_hints.jsonl")
    
    try:
        with open(thread_hints_path, 'w', encoding='utf-8') as f:
            for email_data in parsed_emails:
                hint = {
                    "id": email_data["id"],
                    "message_id": email_data["headers"].get("message_id"),
                    "in_reply_to": email_data["headers"].get("in_reply_to"),
                    "references": email_data["headers"].get("references"),
                    "subject": email_data["headers"].get("subject"),
                    "from": email_data["headers"].get("from"),
                    "date": email_data["headers"].get("date"),
                    "source_filename": email_data["source_filename"]
                }
                f.write(json.dumps(hint, ensure_ascii=False) + '\n')
        logging.info(f"Thread hints saved to {thread_hints_path}")
    except Exception as e:
        logging.error(f"Failed to write thread hints: {e}")
        raise


def main():
    """Main execution function"""
    try:
        # Setup
        data_dir, logs_dir, timestamp = setup_directories_and_logging()
        
        # Load data
        df = load_email_data(data_dir)
        
        # Process emails
        parsed_emails, processing_time = process_all_emails(df)
        
        # Save results
        save_parsed_emails(parsed_emails, data_dir)
        
        # Generate report
        input_csv = os.path.join(data_dir, "emails_sampled_5k.csv")
        generate_parsing_report(parsed_emails, processing_time, input_csv, data_dir, timestamp)
        
        # Extract thread hints
        extract_thread_hints(parsed_emails, data_dir)
        
        logging.info("Email parsing pipeline completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()