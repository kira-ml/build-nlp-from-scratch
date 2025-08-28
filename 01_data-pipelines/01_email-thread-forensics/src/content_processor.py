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

Author: kira-ml (GitHub, machine learning student)
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
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import uuid
from dataclasses import dataclass, asdict

# Configure logging
def setup_logging():
    """Setup comprehensive logging for the content processing pipeline
    
    Creates dual logging to file and console with timestamped filenames.
    Ensures logs directory exists before configuring handlers.
    
    Returns:
        Logger: Configured logger instance for content processing
    """
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

# Transformation Lineage Classes
@dataclass
class TransformationStep:
    """Records a single transformation step with full lineage information"""
    step_id: str
    transformation_type: str
    parameters: Dict[str, Any]
    input_content: str
    output_content: str
    content_changes: Dict[str, Any]
    timestamp: str
    reversibility_metadata: Dict[str, Any]

class TransformationLineageTracker:
    """Tracks and manages transformation lineage for email content processing"""
    
    def __init__(self):
        self.transformations: List[TransformationStep] = []
        self.lineage_id = str(uuid.uuid4())
        
    def record_transformation(
        self,
        transformation_type: str,
        input_content: str,
        output_content: str,
        parameters: Dict[str, Any] = None,
        reversibility_metadata: Dict[str, Any] = None
    ) -> str:
        """Record a transformation step with before/after content snapshots
        
        Args:
            transformation_type: Type of transformation applied
            input_content: Content before transformation
            output_content: Content after transformation
            parameters: Parameters used in transformation
            reversibility_metadata: Metadata needed to reverse the transformation
            
        Returns:
            str: Unique step ID for the transformation
        """
        step_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Calculate content changes
        content_changes = self._calculate_content_changes(input_content, output_content)
        
        # Default parameters and reversibility metadata
        parameters = parameters or {}
        reversibility_metadata = reversibility_metadata or {}
        
        transformation = TransformationStep(
            step_id=step_id,
            transformation_type=transformation_type,
            parameters=parameters,
            input_content=input_content,
            output_content=output_content,
            content_changes=content_changes,
            timestamp=timestamp,
            reversibility_metadata=reversibility_metadata
        )
        
        self.transformations.append(transformation)
        logger.debug(f"Recorded transformation: {transformation_type} (ID: {step_id})")
        
        return step_id
    
    def _calculate_content_changes(self, input_content: str, output_content: str) -> Dict[str, Any]:
        """Calculate detailed changes between input and output content"""
        return {
            'input_length': len(input_content),
            'output_length': len(output_content),
            'length_change': len(output_content) - len(input_content),
            'input_lines': len(input_content.splitlines()),
            'output_lines': len(output_content.splitlines()),
            'character_change_ratio': len(output_content) / len(input_content) if input_content else 0,
            'content_modified': input_content != output_content
        }
    
    def get_lineage_chain(self) -> List[Dict[str, Any]]:
        """Get complete transformation lineage chain"""
        return [asdict(step) for step in self.transformations]
    
    def get_original_content(self) -> str:
        """Get the original content before any transformations"""
        if self.transformations:
            return self.transformations[0].input_content
        return ""
    
    def get_final_content(self) -> str:
        """Get the final content after all transformations"""
        if self.transformations:
            return self.transformations[-1].output_content
        return ""
    
    def reconstruct_from_step(self, step_id: str) -> Optional[str]:
        """Reconstruct content from a specific transformation step
        
        Args:
            step_id: ID of the transformation step to reconstruct from
            
        Returns:
            Content at the specified transformation step, or None if not found
        """
        for step in self.transformations:
            if step.step_id == step_id:
                return step.input_content
        return None

# Configuration
class ProcessorConfig:
    """Configuration settings for content processor
    
    Centralized configuration management for processing parameters
    and file paths to ensure consistency across pipeline components.
    
    Attributes:
        BASE_DIR (Path): Project root directory
        DATA_DIR (Path): Processed data directory path
        INPUT_FILE (str): Source parsed emails filename
        OUTPUT_PROCESSED (str): Processed emails output filename
        OUTPUT_FINGERPRINTS (str): Content fingerprints output filename
        BATCH_SIZE (int): Number of emails to process in each batch
        MAX_ERRORS (int): Maximum allowed processing errors before termination
    """
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
    """Load parsed emails from JSONL file
    
    Reads email data line-by-line to handle large datasets without memory issues.
    Invalid JSON lines are logged and skipped to maintain processing continuity.
    
    Returns:
        List[Dict]: List of parsed email dictionaries
        
    Raises:
        FileNotFoundError: If input file doesn't exist at expected path
        
    Example:
        >>> emails = load_parsed_emails()
        >>> print(f"Loaded {len(emails)} emails")
    """
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

def extract_email_body(email_data: Dict, lineage_tracker: TransformationLineageTracker) -> str:
    """Extract and clean email body from parsed email data with lineage tracking
    
    Prioritizes HTML content when available, falling back to plain text.
    Applies appropriate cleaning based on content type while recording all transformations.
    
    Args:
        email_data (Dict): Parsed email dictionary containing body content
        lineage_tracker (TransformationLineageTracker): Tracker for recording transformations
        
    Returns:
        str: Cleaned email body text, empty string if extraction fails
    """
    try:
        # Get body text from parsed email structure
        body_text = email_data.get('body_text', '')
        body_html = email_data.get('body_html', '')
        
        # Record initial content selection
        if body_html and body_html.strip():
            original_content = body_html
            lineage_tracker.record_transformation(
                transformation_type="content_selection",
                input_content=f"HTML: {body_html}\nTEXT: {body_text}",
                output_content=body_html,
                parameters={"selection_criteria": "html_preferred", "has_html": True, "has_text": bool(body_text)},
                reversibility_metadata={"alternative_text": body_text, "selection_reason": "html_available"}
            )
            return extract_from_html(body_html, lineage_tracker)
        elif body_text and body_text.strip():
            original_content = body_text
            lineage_tracker.record_transformation(
                transformation_type="content_selection",
                input_content=f"HTML: {body_html}\nTEXT: {body_text}",
                output_content=body_text,
                parameters={"selection_criteria": "text_fallback", "has_html": bool(body_html), "has_text": True},
                reversibility_metadata={"alternative_html": body_html, "selection_reason": "html_unavailable"}
            )
            return clean_plain_text(body_text, lineage_tracker)
        else:
            # Record empty content case
            lineage_tracker.record_transformation(
                transformation_type="content_selection",
                input_content=f"HTML: {body_html}\nTEXT: {body_text}",
                output_content="",
                parameters={"selection_criteria": "no_content", "has_html": False, "has_text": False},
                reversibility_metadata={"original_html": body_html, "original_text": body_text}
            )
            logger.warning(f"No body content found for email {email_data.get('id', 'unknown')}")
            return ""
            
    except Exception as e:
        logger.error(f"Error extracting body for email {email_data.get('id', 'unknown')}: {e}")
        # Record error transformation
        lineage_tracker.record_transformation(
            transformation_type="extraction_error",
            input_content=str(email_data),
            output_content="",
            parameters={"error": str(e)},
            reversibility_metadata={"original_data": email_data}
        )
        return ""

def extract_from_html(html_content: str, lineage_tracker: TransformationLineageTracker) -> str:
    """Convert HTML to clean plain text with lineage tracking
    
    Preserves paragraph structure and line breaks while removing HTML tags.
    Handles common HTML email formatting patterns for better readability.
    Records all transformation steps for reversibility.
    
    Args:
        html_content (str): Raw HTML email content
        lineage_tracker (TransformationLineageTracker): Tracker for recording transformations
        
    Returns:
        str: Cleaned plain text representation of HTML content
    """
    try:
        # Record HTML parsing step
        soup = BeautifulSoup(html_content, 'html.parser')
        intermediate_content = str(soup)
        
        lineage_tracker.record_transformation(
            transformation_type="html_parsing",
            input_content=html_content,
            output_content=intermediate_content,
            parameters={"parser": "html.parser", "soup_features": list(soup.find_all())[:10]},
            reversibility_metadata={"original_html": html_content, "parser_used": "html.parser"}
        )
        
        # Record <br> tag replacement
        br_tags_found = len(soup.find_all('br'))
        for br in soup.find_all('br'):
            br.replace_with('\n')
        
        content_after_br = str(soup)
        lineage_tracker.record_transformation(
            transformation_type="br_tag_replacement",
            input_content=intermediate_content,
            output_content=content_after_br,
            parameters={"br_tags_replaced": br_tags_found},
            reversibility_metadata={"br_positions": [str(br) for br in soup.find_all('br')][:10]}
        )
        
        # Record <p> tag replacement
        p_tags_found = len(soup.find_all('p'))
        for p in soup.find_all('p'):
            p.insert_after('\n\n')
        
        content_after_p = str(soup)
        lineage_tracker.record_transformation(
            transformation_type="p_tag_replacement",
            input_content=content_after_br,
            output_content=content_after_p,
            parameters={"p_tags_processed": p_tags_found},
            reversibility_metadata={"p_content": [p.get_text()[:100] for p in soup.find_all('p')][:10]}
        )
        
        # Record text extraction
        text = soup.get_text()
        lineage_tracker.record_transformation(
            transformation_type="html_text_extraction",
            input_content=content_after_p,
            output_content=text,
            parameters={"extraction_method": "beautifulsoup.get_text"},
            reversibility_metadata={"html_structure_preserved": False}
        )
        
        # Clean up extracted text
        return clean_plain_text(text, lineage_tracker)
        
    except Exception as e:
        logger.warning(f"HTML parsing failed, using raw content: {e}")
        # Record fallback to raw content
        lineage_tracker.record_transformation(
            transformation_type="html_parsing_fallback",
            input_content=html_content,
            output_content=html_content,
            parameters={"error": str(e), "fallback_method": "raw_content"},
            reversibility_metadata={"original_html": html_content, "parsing_error": str(e)}
        )
        return clean_plain_text(html_content, lineage_tracker)

def clean_plain_text(text: str, lineage_tracker: TransformationLineageTracker) -> str:
    """Clean and normalize plain text content with lineage tracking
    
    Removes excessive whitespace while preserving intentional formatting.
    Splits content into lines for granular cleaning operations.
    Records all transformations for full reversibility.
    
    Args:
        text (str): Raw plain text content
        lineage_tracker (TransformationLineageTracker): Tracker for recording transformations
        
    Returns:
        str: Cleaned and normalized text content
    """
    if not isinstance(text, str):
        lineage_tracker.record_transformation(
            transformation_type="type_validation",
            input_content=str(text),
            output_content="",
            parameters={"input_type": type(text).__name__, "expected_type": "str"},
            reversibility_metadata={"original_value": text, "original_type": type(text).__name__}
        )
        return ""
    
    original_text = text
    
    # Record line splitting
    lines = text.splitlines()
    lineage_tracker.record_transformation(
        transformation_type="line_splitting",
        input_content=text,
        output_content='\n'.join(lines),
        parameters={"line_count": len(lines), "split_method": "splitlines"},
        reversibility_metadata={"original_line_endings": "preserved"}
    )
    
    # Record line cleaning process
    cleaned_lines = []
    line_changes = []
    
    for i, line in enumerate(lines):
        original_line = line
        # Remove excessive whitespace but preserve intentional formatting
        cleaned_line = re.sub(r'\s+', ' ', line.strip())
        
        if cleaned_line:  # Skip empty lines
            cleaned_lines.append(cleaned_line)
            if original_line != cleaned_line:
                line_changes.append({
                    'line_number': i,
                    'original': original_line,
                    'cleaned': cleaned_line,
                    'changes': 'whitespace_normalized'
                })
        else:
            line_changes.append({
                'line_number': i,
                'original': original_line,
                'cleaned': '',
                'changes': 'empty_line_removed'
            })
    
    final_text = '\n'.join(cleaned_lines)
    
    # Record final cleaning transformation
    lineage_tracker.record_transformation(
        transformation_type="text_cleaning",
        input_content=original_text,
        output_content=final_text,
        parameters={
            "original_lines": len(lines),
            "cleaned_lines": len(cleaned_lines),
            "empty_lines_removed": len(lines) - len(cleaned_lines),
            "whitespace_pattern": r'\s+',
            "replacement_pattern": ' '
        },
        reversibility_metadata={
            "line_changes": line_changes[:50],  # Limit to first 50 changes for performance
            "original_length": len(original_text),
            "cleaning_rules": ["strip_lines", "normalize_whitespace", "remove_empty_lines"]
        }
    )
    
    return final_text

def detect_quoted_content(text_body: str, lineage_tracker: TransformationLineageTracker) -> Dict[str, str]:
    """Detect and separate quoted content from original content with lineage tracking
    
    Identifies quoted text using common email reply patterns and separates
    it from original message content for cleaner analysis.
    Records all pattern matching and content separation decisions.
    
    Args:
        text_body (str): Full email body text to analyze
        lineage_tracker (TransformationLineageTracker): Tracker for recording transformations
        
    Returns:
        Dict[str, str]: Dictionary with 'original' and 'quoted' content sections
    """
    if not isinstance(text_body, str) or not text_body.strip():
        lineage_tracker.record_transformation(
            transformation_type="quote_detection_validation",
            input_content=str(text_body),
            output_content="{'original': '', 'quoted': ''}",
            parameters={"validation_result": "empty_or_invalid_input"},
            reversibility_metadata={"original_input": text_body, "input_type": type(text_body).__name__}
        )
        return {'original': '', 'quoted': ''}
    
    quote_lines = []
    original_lines = []
    lines = text_body.splitlines()
    in_quote = False
    quote_triggers = []
    
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
    
    # Record pattern matching process
    for i, line in enumerate(lines):
        line_quote_matches = []
        
        # Check if this line indicates start of quoted content
        if not in_quote:
            for pattern in quote_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    in_quote = True
                    line_quote_matches.append(pattern)
                    quote_triggers.append({
                        'line_number': i,
                        'line_content': line,
                        'matching_pattern': pattern,
                        'quote_started': True
                    })
                    break
        
        # Categorize the line
        if in_quote:
            quote_lines.append(line)
        else:
            original_lines.append(line)
    
    original_text = '\n'.join(original_lines).strip()
    quoted_text = '\n'.join(quote_lines).strip()
    
    # Record quote detection transformation
    lineage_tracker.record_transformation(
        transformation_type="quote_content_detection",
        input_content=text_body,
        output_content=f"ORIGINAL:\n{original_text}\n\nQUOTED:\n{quoted_text}",
        parameters={
            "total_lines": len(lines),
            "original_lines": len(original_lines),
            "quoted_lines": len(quote_lines),
            "quote_patterns_used": quote_patterns,
            "quote_triggers_found": len(quote_triggers)
        },
        reversibility_metadata={
            "line_categorization": [
                {"line_number": i, "content": line[:100], "category": "quoted" if i >= len(original_lines) else "original"}
                for i, line in enumerate(lines[:50])  # Limit for performance
            ],
            "quote_triggers": quote_triggers,
            "patterns_matched": [trigger['matching_pattern'] for trigger in quote_triggers],
            "reconstruction_method": "line_by_line_categorization"
        }
    )
    
    return {
        'original': original_text,
        'quoted': quoted_text
    }

def normalize_headers(email_data: Dict) -> Dict[str, Optional[str]]:
    """Normalize and standardize email headers
    
    Processes standard email headers to create consistent, comparable values.
    Handles address parsing, date standardization, and case normalization.
    
    Args:
        email_data (Dict): Parsed email dictionary containing raw headers
        
    Returns:
        Dict[str, Optional[str]]: Normalized header values with standardized formats
    """
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
    """Generate SHA-256 hash for content deduplication
    
    Creates deterministic hash based on key email content fields to
    enable efficient duplicate detection and content fingerprinting.
    
    Args:
        content_dict (Dict[str, str]): Dictionary of content fields to hash
        
    Returns:
        str: SHA-256 hash hexadecimal string of content
    """
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
    """Process a single email and return processed email and fingerprint with lineage tracking
    
    Orchestrates the complete processing workflow for one email including
    content extraction, cleaning, normalization, and fingerprinting.
    Records complete transformation lineage for forensic analysis.
    
    Args:
        email_data (Dict): Raw parsed email data to process
        
    Returns:
        Tuple[Optional[Dict], Optional[Dict]]: Tuple of (processed_email, fingerprint)
            Returns (None, None) if processing fails
    """
    try:
        email_id = email_data.get('id', 'unknown')
        
        # Initialize lineage tracker for this email
        lineage_tracker = TransformationLineageTracker()
        
        # Record initial email data state
        lineage_tracker.record_transformation(
            transformation_type="email_processing_start",
            input_content=json.dumps(email_data, default=str),
            output_content=json.dumps(email_data, default=str),
            parameters={"email_id": email_id, "processing_stage": "initialization"},
            reversibility_metadata={"original_email_data": email_data}
        )
        
        # Extract and clean body content
        plain_text = extract_email_body(email_data, lineage_tracker)
        if not plain_text:
            logger.warning(f"No extractable content for email {email_id}")
            return None, None
        
        # Detect quoted content
        content_parts = detect_quoted_content(plain_text, lineage_tracker)
        
        # Normalize headers
        norm_headers = normalize_headers(email_data)
        
        # Record header normalization
        lineage_tracker.record_transformation(
            transformation_type="header_normalization",
            input_content=json.dumps(email_data.get('headers', {}), default=str),
            output_content=json.dumps(norm_headers, default=str),
            parameters={"normalization_fields": list(norm_headers.keys())},
            reversibility_metadata={"original_headers": email_data.get('headers', {})}
        )
        
        # Generate content hash
        hash_input = {
            'from': norm_headers['from'],
            'to': norm_headers['to'],
            'subject': norm_headers['subject'],
            'original_body': content_parts['original'],
            'date': norm_headers['date']
        }
        content_hash = generate_content_hash(hash_input)
        
        # Record hash generation
        lineage_tracker.record_transformation(
            transformation_type="content_hash_generation",
            input_content=json.dumps(hash_input, default=str),
            output_content=content_hash,
            parameters={"hash_algorithm": "sha256", "hash_fields": list(hash_input.keys())},
            reversibility_metadata={"hash_input_data": hash_input}
        )
        
        # Get complete lineage chain
        transformation_lineage = lineage_tracker.get_lineage_chain()
        
        # Create processed email record with lineage
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
            'transformation_lineage': {
                'lineage_id': lineage_tracker.lineage_id,
                'transformation_count': len(transformation_lineage),
                'transformation_chain': transformation_lineage,
                'original_content': lineage_tracker.get_original_content(),
                'final_content': lineage_tracker.get_final_content()
            },
            'processing_metadata': {
                'html_stripped': bool(email_data.get('body_html')),
                'quotes_detected': bool(content_parts['quoted']),
                'timestamp_parsed': bool(norm_headers['date']),
                'processed_at': datetime.now().isoformat(),
                'original_length': len(lineage_tracker.get_original_content()),
                'processed_length': len(content_parts['original']),
                'lineage_tracking_enabled': True,
                'transformation_steps': len(transformation_lineage)
            }
        }
        
        # Create fingerprint record
        fingerprint = {
            'id': email_id,
            'content_hash': content_hash,
            'message_id': norm_headers['message_id'],
            'lineage_id': lineage_tracker.lineage_id
        }
        
        # Record processing completion
        lineage_tracker.record_transformation(
            transformation_type="email_processing_complete",
            input_content=json.dumps(email_data, default=str),
            output_content=json.dumps(processed_email, default=str),
            parameters={"processing_status": "success", "final_email_id": email_id},
            reversibility_metadata={"complete_processing_chain": True}
        )
        
        return processed_email, fingerprint
        
    except Exception as e:
        logger.error(f"Failed to process email {email_data.get('id', 'unknown')}: {e}")
        return None, None

def process_all_emails() -> Tuple[List[Dict], List[Dict], Dict]:
    """Process all emails and return results with statistics
    
    Executes batch processing of all emails with memory-efficient streaming.
    Tracks processing metrics and handles error thresholds gracefully.
    
    Returns:
        Tuple[List[Dict], List[Dict], Dict]: Tuple of (processed_emails, fingerprints, stats)
    """
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
    """Save processing results to files with lineage data
    
    Persists processed emails, content fingerprints, and processing statistics
    to JSONL and JSON files respectively with UTF-8 encoding support.
    Includes transformation lineage data for forensic analysis.
    
    Args:
        processed_emails (List[Dict]): List of fully processed email records with lineage
        fingerprints (List[Dict]): List of content fingerprint records
        stats (Dict): Processing statistics dictionary
    """
    # Ensure output directory exists
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save processed emails with lineage
    output_processed = config.DATA_DIR / config.OUTPUT_PROCESSED
    with open(output_processed, 'w', encoding='utf-8') as f:
        for email in processed_emails:
            f.write(json.dumps(email, ensure_ascii=False, default=str) + '\n')
    logger.info(f"Saved {len(processed_emails)} processed emails with lineage to {output_processed}")
    
    # Save content fingerprints
    output_fingerprints = config.DATA_DIR / config.OUTPUT_FINGERPRINTS
    with open(output_fingerprints, 'w', encoding='utf-8') as f:
        for fingerprint in fingerprints:
            f.write(json.dumps(fingerprint, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(fingerprints)} content fingerprints to {output_fingerprints}")
    
    # Save separate lineage summary for quick access
    lineage_summary_path = config.DATA_DIR / "transformation_lineage_summary.jsonl"
    with open(lineage_summary_path, 'w', encoding='utf-8') as f:
        for email in processed_emails:
            if 'transformation_lineage' in email:
                lineage_summary = {
                    'email_id': email['id'],
                    'lineage_id': email['transformation_lineage']['lineage_id'],
                    'transformation_count': email['transformation_lineage']['transformation_count'],
                    'transformations': [
                        {
                            'step_id': step['step_id'],
                            'type': step['transformation_type'],
                            'timestamp': step['timestamp']
                        }
                        for step in email['transformation_lineage']['transformation_chain']
                    ]
                }
                f.write(json.dumps(lineage_summary, ensure_ascii=False) + '\n')
    logger.info(f"Saved transformation lineage summary to {lineage_summary_path}")
    
    # Enhance stats with lineage information
    lineage_stats = {
        'emails_with_lineage': sum(1 for email in processed_emails if 'transformation_lineage' in email),
        'total_transformations': sum(
            email.get('transformation_lineage', {}).get('transformation_count', 0)
            for email in processed_emails
        ),
        'avg_transformations_per_email': 0
    }
    
    if processed_emails:
        lineage_stats['avg_transformations_per_email'] = (
            lineage_stats['total_transformations'] / len(processed_emails)
        )
    
    stats.update({
        'lineage_tracking': lineage_stats,
        'lineage_enabled': True
    })
    
    # Save enhanced processing report
    report_path = config.DATA_DIR / "content_processing_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved enhanced processing report with lineage stats to {report_path}")

def validate_outputs():
    """Validate that output files were created correctly including lineage data
    
    Performs basic validation checks on output files to ensure
    successful completion and non-empty results. Includes validation
    of transformation lineage files.
    
    Returns:
        bool: True if all validations pass, False otherwise
    """
    output_files = [
        config.DATA_DIR / config.OUTPUT_PROCESSED,
        config.DATA_DIR / config.OUTPUT_FINGERPRINTS,
        config.DATA_DIR / "content_processing_report.json",
        config.DATA_DIR / "transformation_lineage_summary.jsonl"
    ]
    
    for file_path in output_files:
        if not file_path.exists():
            logger.error(f"Output file missing: {file_path}")
            return False
        
        if file_path.stat().st_size == 0:
            logger.error(f"Output file is empty: {file_path}")
            return False
    
    # Additional validation for lineage data structure
    try:
        with open(config.DATA_DIR / config.OUTPUT_PROCESSED, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                email_data = json.loads(first_line)
                if 'transformation_lineage' not in email_data:
                    logger.warning("Processed emails missing transformation lineage data")
                    return False
                
                if 'lineage_id' not in email_data['transformation_lineage']:
                    logger.error("Invalid lineage structure - missing lineage_id")
                    return False
    except Exception as e:
        logger.error(f"Failed to validate lineage structure: {e}")
        return False
    
    logger.info("All output files and lineage data validated successfully")
    return True

def main():
    """Main execution function
    
    Entry point for the complete email content processing pipeline.
    Orchestrates all processing stages and handles top-level errors.
    """
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
            logger.info("Content processing pipeline with lineage tracking completed successfully!")
            print(f"\n✅ PROCESSING COMPLETE WITH LINEAGE TRACKING!")
            print(f"📊 Processed: {stats['successfully_processed']}/{stats['total_input']} emails")
            print(f"📈 Success Rate: {stats['success_rate']:.1%}")
            print(f"💬 Quotes Detected: {stats['quotes_detected']} emails")
            print(f"🌐 HTML Emails: {stats['html_emails']} emails")
            if 'lineage_tracking' in stats:
                print(f"� Lineage Tracked: {stats['lineage_tracking']['emails_with_lineage']} emails")
                print(f"⚡ Total Transformations: {stats['lineage_tracking']['total_transformations']}")
                print(f"📝 Avg Transformations/Email: {stats['lineage_tracking']['avg_transformations_per_email']:.1f}")
            print(f"�📁 Output Files: {config.DATA_DIR}")
            print(f"📋 Lineage Summary: transformation_lineage_summary.jsonl")
        else:
            raise Exception("Output validation failed")
            
    except Exception as e:
        logger.error(f"Content processing pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()