"""
Email Content Processing Pipeline

This module implements comprehensive email content cleaning and normalization
for forensic analysis. It processes parsed emails to extract clean content,
detect quoted text, normalize headers, and generate content fingerprints.

Key Features:
- HTML to plain text conversion with formatting preservation
- Quoted content detection using regex patterns
- Email header normalization and validation
- Content fingerprinting for deduplication
- Robust error handling and logging
- Progress tracking for large datasets

Author: Email Forensics Pipeline
Version: 1.0
"""

import pandas as pd
import os
import re
import json
import email.utils
import hashlib
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure logging
def setup_logging():
    """Setup comprehensive logging for the content processing pipeline"""
    BASE_DIR = Path(__file__).parent.parent
    LOGS_DIR = BASE_DIR / "logs"
    LOGS_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"content_processor_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logging
logger = setup_logging()

# Configuration
class ProcessorConfig:
    """Configuration settings for content processor"""
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / "data" / "processed"
        self.INPUT_FILE = "emails_parsed.jsonl"  # Use parsed emails, not CSV
        self.OUTPUT_PROCESSED = "emails_processed.jsonl"
        self.OUTPUT_FINGERPRINTS = "content_fingerprints.jsonl"
        self.BATCH_SIZE = 100
        self.MAX_ERRORS = 50

config = ProcessorConfig()

def load_parsed_emails() -> List[Dict]:
    """Load parsed emails from JSONL file"""
    input_path = config.DATA_DIR / config.INPUT_FILE
    
    if not input_path.exists():
        raise FileNotFoundError(f"Parsed emails file not found: {input_path}")
    
    emails = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                email_data = json.loads(line.strip())
                emails.append(email_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(emails)} parsed emails from {input_path}")
    return emails

def extract_email_body(email_data: Dict) -> str:
    """Extract and clean email body from parsed email data"""
    try:
        # Get body text from parsed email structure
        body_text = email_data.get('body_text', '')
        body_html = email_data.get('body_html', '')
        
        # Prefer HTML if available, fallback to plain text
        if body_html and body_html.strip():
            return extract_from_html(body_html)
        elif body_text and body_text.strip():
            return clean_plain_text(body_text)
        else:
            logger.warning(f"No body content found for email {email_data.get('id', 'unknown')}")
            return ""
            
    except Exception as e:
        logger.error(f"Error extracting body for email {email_data.get('id', 'unknown')}: {e}")
        return ""

def extract_from_html(html_content: str) -> str:
    """Convert HTML to clean plain text"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Replace <br> tags with newlines
        for br in soup.find_all('br'):
            br.replace_with('\n')
        
        # Replace <p> tags with double newlines for paragraph separation
        for p in soup.find_all('p'):
            p.insert_after('\n\n')
        
        # Get text and clean up extra whitespace
        text = soup.get_text()
        return clean_plain_text(text)
        
    except Exception as e:
        logger.warning(f"HTML parsing failed, using raw content: {e}")
        return clean_plain_text(html_content)

def clean_plain_text(text: str) -> str:
    """Clean and normalize plain text content"""
    if not isinstance(text, str):
        return ""
    
    # Split into lines and clean each line
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        # Remove excessive whitespace but preserve intentional formatting
        cleaned_line = re.sub(r'\s+', ' ', line.strip())
        if cleaned_line:  # Skip empty lines
            cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)

def detect_quoted_content(text_body: str) -> Dict[str, str]:
    """Detect and separate quoted content from original content"""
    if not isinstance(text_body, str) or not text_body.strip():
        return {'original': '', 'quoted': ''}
    
    quote_lines = []
    original_lines = []
    lines = text_body.splitlines()
    in_quote = False
    
    # Patterns that indicate start of quoted content
    quote_patterns = [
        r'^\s*>',  # Lines starting with >
        r'On\s+.+\s+wrote:',  # "On [date/time] [person] wrote:"
        r'From:\s+.+',  # Forward headers
        r'Sent:\s+.+',  # Outlook-style headers
        r'To:\s+.+',  # Email headers
        r'Subject:\s+.+',  # Subject lines in forwards
        r'-----Original Message-----',  # Outlook original message
        r'________________________________',  # Outlook separator
    ]
    
    for line in lines:
        # Check if this line indicates start of quoted content
        if not in_quote:
            for pattern in quote_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    in_quote = True
                    break
        
        # Categorize the line
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

def normalize_headers(email_data: Dict) -> Dict[str, Optional[str]]:
    """Normalize and standardize email headers"""
    try:
        headers = email_data.get('headers', {})
        
        # Normalize from address
        raw_from = headers.get('from', '')
        try:
            from_name, from_addr = email.utils.parseaddr(raw_from)
            if from_addr:
                cleaned_from = f"{from_name.strip()} <{from_addr.lower().strip()}>" if from_name else from_addr.lower().strip()
            else:
                cleaned_from = raw_from.strip()
        except Exception:
            cleaned_from = raw_from.strip() if raw_from else ""
        
        # Normalize to addresses
        raw_to = headers.get('to', '')
        to_list = []
        if isinstance(raw_to, str) and raw_to:
            for addr in raw_to.split(','):
                try:
                    name, addr_clean = email.utils.parseaddr(addr.strip())
                    if addr_clean:
                        formatted = f"{name.strip()} <{addr_clean.lower()}>" if name else addr_clean.lower()
                        to_list.append(formatted)
                except Exception:
                    continue
        
        cleaned_to = ', '.join(to_list)
        
        # Normalize date
        raw_date = headers.get('date', '')
        standardized_date = None
        if raw_date:
            try:
                dt = email.utils.parsedate_to_datetime(raw_date)
                standardized_date = dt.isoformat()
            except Exception:
                logger.warning(f"Failed to parse date: {raw_date}")
        
        return {
            'from': cleaned_from,
            'to': cleaned_to,
            'date': standardized_date,
            'subject': headers.get('subject', '').strip(),
            'message_id': headers.get('message_id', '').strip()
        }
        
    except Exception as e:
        logger.error(f"Header normalization failed for email {email_data.get('id', 'unknown')}: {e}")
        return {
            'from': '',
            'to': '',
            'date': None,
            'subject': '',
            'message_id': ''
        }

def generate_content_hash(content_dict: Dict[str, str]) -> str:
    """Generate SHA-256 hash for content deduplication"""
    try:
        # Create deterministic string for hashing
        hash_components = [
            content_dict.get('from', '').lower().strip(),
            content_dict.get('to', '').lower().strip(),
            content_dict.get('subject', '').strip(),
            content_dict.get('original_body', '').strip(),
            content_dict.get('date', '') or ""
        ]
        
        concat_string = '|'.join(hash_components)
        hash_obj = hashlib.sha256(concat_string.encode('utf-8'))
        return hash_obj.hexdigest()
        
    except Exception as e:
        logger.error(f"Hash generation failed: {e}")
        return hashlib.sha256(str(datetime.now()).encode()).hexdigest()

def process_single_email(email_data: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Process a single email and return processed email and fingerprint"""
    try:
        email_id = email_data.get('id', 'unknown')
        
        # Extract and clean body content
        plain_text = extract_email_body(email_data)
        if not plain_text:
            logger.warning(f"No extractable content for email {email_id}")
            return None, None
        
        # Detect quoted content
        content_parts = detect_quoted_content(plain_text)
        
        # Normalize headers
        norm_headers = normalize_headers(email_data)
        
        # Generate content hash
        hash_input = {
            'from': norm_headers['from'],
            'to': norm_headers['to'],
            'subject': norm_headers['subject'],
            'original_body': content_parts['original'],
            'date': norm_headers['date']
        }
        content_hash = generate_content_hash(hash_input)
        
        # Create processed email record
        processed_email = {
            'id': email_id,
            'message_id': norm_headers['message_id'],
            'from': norm_headers['from'],
            'to': norm_headers['to'],
            'date': norm_headers['date'],
            'subject': norm_headers['subject'],
            'body_original': content_parts['original'],
            'body_quoted': content_parts['quoted'],
            'content_hash': content_hash,
            'source_filename': email_data.get('source_filename', ''),
            'processing_metadata': {
                'html_stripped': bool(email_data.get('body_html')),
                'quotes_detected': bool(content_parts['quoted']),
                'timestamp_parsed': bool(norm_headers['date']),
                'processed_at': datetime.now().isoformat(),
                'original_length': len(plain_text),
                'processed_length': len(content_parts['original'])
            }
        }
        
        # Create fingerprint record
        fingerprint = {
            'id': email_id,
            'content_hash': content_hash,
            'message_id': norm_headers['message_id']
        }
        
        return processed_email, fingerprint
        
    except Exception as e:
        logger.error(f"Failed to process email {email_data.get('id', 'unknown')}: {e}")
        return None, None

def process_all_emails() -> Tuple[List[Dict], List[Dict], Dict]:
    """Process all emails and return results with statistics"""
    logger.info("Starting email content processing pipeline...")
    
    # Load parsed emails
    emails = load_parsed_emails()
    total_emails = len(emails)
    
    processed_emails = []
    content_fingerprints = []
    error_count = 0
    
    # Process in batches for memory efficiency
    for i in range(0, total_emails, config.BATCH_SIZE):
        batch = emails[i:i + config.BATCH_SIZE]
        batch_num = i // config.BATCH_SIZE + 1
        total_batches = (total_emails + config.BATCH_SIZE - 1) // config.BATCH_SIZE
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} emails)")
        
        for email_data in batch:
            processed, fingerprint = process_single_email(email_data)
            
            if processed and fingerprint:
                processed_emails.append(processed)
                content_fingerprints.append(fingerprint)
            else:
                error_count += 1
                if error_count > config.MAX_ERRORS:
                    logger.error(f"Too many errors ({error_count}), stopping processing")
                    break
    
    # Generate processing statistics
    stats = {
        'total_input': total_emails,
        'successfully_processed': len(processed_emails),
        'failed_processing': error_count,
        'success_rate': len(processed_emails) / total_emails if total_emails > 0 else 0,
        'quotes_detected': sum(1 for e in processed_emails if e['processing_metadata']['quotes_detected']),
        'html_emails': sum(1 for e in processed_emails if e['processing_metadata']['html_stripped']),
        'processing_timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Processing complete: {stats['successfully_processed']}/{stats['total_input']} emails processed successfully")
    
    return processed_emails, content_fingerprints, stats

def save_results(processed_emails: List[Dict], fingerprints: List[Dict], stats: Dict):
    """Save processing results to files"""
    # Ensure output directory exists
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save processed emails
    output_processed = config.DATA_DIR / config.OUTPUT_PROCESSED
    with open(output_processed, 'w', encoding='utf-8') as f:
        for email in processed_emails:
            f.write(json.dumps(email, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(processed_emails)} processed emails to {output_processed}")
    
    # Save content fingerprints
    output_fingerprints = config.DATA_DIR / config.OUTPUT_FINGERPRINTS
    with open(output_fingerprints, 'w', encoding='utf-8') as f:
        for fingerprint in fingerprints:
            f.write(json.dumps(fingerprint, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(fingerprints)} content fingerprints to {output_fingerprints}")
    
    # Save processing report
    report_path = config.DATA_DIR / "content_processing_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved processing report to {report_path}")

def validate_outputs():
    """Validate that output files were created correctly"""
    output_files = [
        config.DATA_DIR / config.OUTPUT_PROCESSED,
        config.DATA_DIR / config.OUTPUT_FINGERPRINTS,
        config.DATA_DIR / "content_processing_report.json"
    ]
    
    for file_path in output_files:
        if not file_path.exists():
            logger.error(f"Output file missing: {file_path}")
            return False
        
        if file_path.stat().st_size == 0:
            logger.error(f"Output file is empty: {file_path}")
            return False
    
    logger.info("All output files validated successfully")
    return True

def main():
    """Main execution function"""
    try:
        logger.info("="*60)
        logger.info("EMAIL CONTENT PROCESSING PIPELINE STARTED")
        logger.info("="*60)
        
        # Process all emails
        processed_emails, fingerprints, stats = process_all_emails()
        
        # Save results
        save_results(processed_emails, fingerprints, stats)
        
        # Validate outputs
        if validate_outputs():
            logger.info("Content processing pipeline completed successfully!")
            print(f"\n✅ PROCESSING COMPLETE!")
            print(f"📊 Processed: {stats['successfully_processed']}/{stats['total_input']} emails")
            print(f"📈 Success Rate: {stats['success_rate']:.1%}")
            print(f"💬 Quotes Detected: {stats['quotes_detected']} emails")
            print(f"🌐 HTML Emails: {stats['html_emails']} emails")
            print(f"📁 Output Files: {config.DATA_DIR}")
        else:
            raise Exception("Output validation failed")
            
    except Exception as e:
        logger.error(f"Content processing pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()