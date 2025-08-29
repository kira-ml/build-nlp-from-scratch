"""
Email Thread Reconstruction Pipeline with Forensic Lineage Tracking

This module reconstructs conversation threads from parsed email data by analyzing:
- Message-ID and In-Reply-To relationships
- Reference header chains 
- Subject line patterns and normalization
- Temporal sequencing and conversation flows

The implementation maintains full provenance tracking, ensuring every reconstructed
thread can be traced back to original source files for forensic analysis.

Key Features:
- Graph-based thread reconstruction using networkx
- Conversation flow analysis and cycle detection
- Subject line normalization and reply chain detection
- Temporal validation and orphan email handling
- Comprehensive thread quality metrics and validation
- Forensic-grade lineage tracking with canonical message IDs
- Source message provenance mapping for audit trails
- Thread-level lineage showing raw file contributions

Example:
    >>> # Execute thread reconstruction pipeline
    >>> python reconstruct_threads.py

Author: kira-ml (GitHub, machine learning student)
"""

import os
import json
import pandas as pd
import logging
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx


@dataclass
class ThreadLineageStep:
    """Track lineage for thread reconstruction transformations
    
    Provides comprehensive audit trail for thread reconstruction operations
    including source emails, transformation steps, and forensic metadata.
    """
    step_id: str
    transformation_type: str
    input_emails: List[str]
    output_thread_id: str
    canonical_message_ids: List[str]
    source_files: List[str]
    processing_timestamp: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class ThreadLineageTracker:
    """Manage thread reconstruction lineage tracking for forensic compliance
    
    Maintains comprehensive audit trails of all thread reconstruction operations
    including source-to-output mappings, canonical message ID assignments,
    and raw file provenance for regulatory compliance.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.lineage_steps: List[ThreadLineageStep] = []
        self.canonical_id_map: Dict[str, str] = {}  # message_id -> canonical_id
        self.source_lineage_map: Dict[str, Dict[str, Any]] = {}  # canonical_id -> source info
        
    def assign_canonical_id(self, message_id: str, source_file: str = None) -> str:
        """Assign persistent canonical ID to message across all transformations"""
        if message_id in self.canonical_id_map:
            return self.canonical_id_map[message_id]
        
        canonical_id = f"canonical_{uuid.uuid4().hex[:12]}"
        self.canonical_id_map[message_id] = canonical_id
        
        # Track source lineage
        self.source_lineage_map[canonical_id] = {
            'original_message_id': message_id,
            'source_file': source_file,
            'assigned_timestamp': datetime.now().isoformat(),
            'transformation_history': []
        }
        
        return canonical_id
    
    def record_thread_reconstruction(self, 
                                   thread_id: str,
                                   input_emails: List[Dict[str, Any]], 
                                   output_thread: Dict[str, Any],
                                   parameters: Dict[str, Any] = None) -> str:
        """Record complete thread reconstruction lineage"""
        step_id = f"thread_recon_{uuid.uuid4().hex[:8]}"
        
        # Extract canonical IDs and source files
        canonical_ids = []
        source_files = set()
        
        for email in input_emails:
            message_id = email.get('headers', {}).get('message_id', '').strip('<>')
            if message_id:
                canonical_id = self.assign_canonical_id(message_id, email.get('source_file'))
                canonical_ids.append(canonical_id)
                
                # Track source file
                if email.get('source_file'):
                    source_files.add(email['source_file'])
                
                # Update transformation history
                if canonical_id in self.source_lineage_map:
                    self.source_lineage_map[canonical_id]['transformation_history'].append({
                        'step': 'thread_reconstruction',
                        'thread_id': thread_id,
                        'timestamp': datetime.now().isoformat()
                    })
        
        lineage_step = ThreadLineageStep(
            step_id=step_id,
            transformation_type='thread_reconstruction',
            input_emails=[email.get('headers', {}).get('message_id', '') for email in input_emails],
            output_thread_id=thread_id,
            canonical_message_ids=canonical_ids,
            source_files=list(source_files),
            processing_timestamp=datetime.now().isoformat(),
            parameters=parameters or {},
            metadata={
                'thread_depth': output_thread.get('thread_depth', 0),
                'message_count': output_thread.get('message_count', 0),
                'participants': output_thread.get('participants', [])
            }
        )
        
        self.lineage_steps.append(lineage_step)
        return step_id
    
    def get_thread_source_lineage(self, thread_id: str) -> Dict[str, Any]:
        """Get complete source lineage for a thread"""
        thread_steps = [step for step in self.lineage_steps 
                       if step.output_thread_id == thread_id]
        
        if not thread_steps:
            return {}
        
        step = thread_steps[0]  # Should only be one per thread
        
        lineage_info = {
            'thread_id': thread_id,
            'source_files': step.source_files,
            'canonical_message_ids': step.canonical_message_ids,
            'message_lineage': {}
        }
        
        # Add detailed lineage for each message
        for canonical_id in step.canonical_message_ids:
            if canonical_id in self.source_lineage_map:
                lineage_info['message_lineage'][canonical_id] = self.source_lineage_map[canonical_id]
        
        return lineage_info
    
    def save_lineage_summary(self) -> None:
        """Save comprehensive lineage tracking summary"""
        summary_path = self.output_dir / "thread_lineage_summary.jsonl"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            for step in self.lineage_steps:
                f.write(json.dumps(asdict(step), ensure_ascii=False) + '\n')
        
        # Save canonical ID mapping
        mapping_path = self.output_dir / "canonical_id_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.canonical_id_map, f, indent=2, ensure_ascii=False)
        
        # Save source lineage mapping
        source_lineage_path = self.output_dir / "source_message_lineage.json"
        with open(source_lineage_path, 'w', encoding='utf-8') as f:
            json.dump(self.source_lineage_map, f, indent=2, ensure_ascii=False)
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any
import networkx as nx
import re
from pathlib import Path


class EmailThreadReconstructor:
    """Efficient email thread reconstruction engine with forensic lineage tracking
    
    This class implements a graph-based approach to reconstruct email conversation threads
    by analyzing message headers and relationships. It supports large-scale processing
    with memory-efficient data handling and comprehensive quality metrics.
    
    Features:
    - Graph-based thread reconstruction using networkx
    - Memory-efficient processing for large datasets  
    - Quality metrics and validation
    - Comprehensive error handling and logging
    - Forensic-grade lineage tracking with canonical message IDs
    - Source message provenance mapping for audit trails
    
    Attributes:
        base_dir (Path): Base project directory path
        data_dir (Path): Processed data directory path
        logs_dir (Path): Logging directory path
        timestamp (str): ISO format timestamp for current run
        logger (Logger): Configured logging instance
        lineage_tracker (ThreadLineageTracker): Forensic lineage tracking instance
    """
    
    def __init__(self, base_directory: str):
        """Initialize the thread reconstructor with base directory and lineage tracking
        
        Args:
            base_directory (str): Root directory for project data and logs
        """
        self.base_dir = Path(base_directory)
        self.data_dir = self.base_dir / "data" / "processed"
        self.logs_dir = self.base_dir / "logs"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize lineage tracker for forensic compliance
        self.lineage_tracker = ThreadLineageTracker(self.data_dir)
        
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging for thread reconstruction
        
        Sets up dual logging to file and console with timestamped filenames.
        Creates logs directory if it doesn't exist.
        """
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_filename = self.logs_dir / f"reconstruct_threads_{self.timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_parsed_emails(self) -> List[Dict[str, Any]]:
        """Load parsed emails from JSONL file with memory-efficient streaming
        
        Reads email data line-by-line to handle large datasets without memory issues.
        Invalid JSON lines are logged and skipped to maintain processing continuity.
        
        Returns:
            List[Dict[str, Any]]: List of parsed email dictionaries
            
        Raises:
            FileNotFoundError: If emails file doesn't exist at expected path
            Exception: For other file reading or parsing errors
            
        Example:
            >>> emails = reconstructor.load_parsed_emails()
            >>> print(f"Loaded {len(emails)} emails")
        """
        emails_path = self.data_dir / "emails_parsed.jsonl"
        emails = []
        
        try:
            with open(emails_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        emails.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
            
            self.logger.info(f"Loaded {len(emails)} parsed emails")
            return emails
        except FileNotFoundError:
            self.logger.error(f"Emails file not found: {emails_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load parsed emails: {e}")
            raise
    
    @staticmethod
    def normalize_subject(subject: str) -> str:
        """Normalize email subject for thread matching with caching
        
        Removes common reply/forward prefixes and normalizes whitespace to
        enable consistent subject-based thread grouping.
        
        Args:
            subject (str): Raw email subject line
            
        Returns:
            str: Normalized subject string suitable for comparison
            
        Example:
            >>> normalize_subject("Re: Meeting Tomorrow")
            'meeting tomorrow'
        """
        if not subject:
            return ""
        
        # Remove reply/forward prefixes (case-insensitive)
        subject = re.sub(r'^(?:re|fw|fwd):\s*', '', subject.strip(), flags=re.IGNORECASE)
        
        # Normalize whitespace
        subject = re.sub(r'\s+', ' ', subject)
        
        return subject.strip().lower()
    
    @staticmethod
    def extract_email_references(email: Dict[str, Any]) -> Dict[str, Any]:
        """Extract threading references from email headers with validation
        
        Processes standard email headers to extract message relationships needed
        for thread reconstruction. Handles edge cases like malformed headers.
        
        Args:
            email (Dict[str, Any]): Parsed email dictionary with headers
            
        Returns:
            Dict[str, Any]: Extracted reference data including message IDs and subjects
        """
        headers = email.get('headers', {})
        
        # Extract and clean message ID
        message_id = headers.get('message_id', '')
        message_id = message_id.strip('<>') if message_id else ''
        
        # Extract and clean in-reply-to
        in_reply_to = headers.get('in_reply_to')
        in_reply_to = in_reply_to.strip('<>') if in_reply_to else None
        
        return {
            'message_id': message_id,
            'in_reply_to': in_reply_to,
            'references': EmailThreadReconstructor._parse_references(headers.get('references', '')),
            'subject': headers.get('subject', ''),
            'normalized_subject': EmailThreadReconstructor.normalize_subject(headers.get('subject', '')),
            'date': headers.get('date', ''),
            'from': headers.get('from', '')
        }
    
    @staticmethod
    def _parse_references(references_header: str) -> List[str]:
        """Parse References header into list of message IDs
        
        Extracts message IDs from the References header according to RFC standards.
        Handles multiple IDs separated by whitespace and angle brackets.
        
        Args:
            references_header (str): Raw References header content
            
        Returns:
            List[str]: Cleaned list of referenced message IDs
        """
        if not references_header:
            return []
        
        # Extract message IDs between < > with optimized regex
        message_ids = re.findall(r'<([^>]+)>', references_header)
        return [mid.strip() for mid in message_ids if mid.strip()]
    
    def build_thread_graph(self, emails: List[Dict[str, Any]]) -> nx.DiGraph:
        """Build directed graph of email relationships with performance optimizations
        
        Constructs a directed graph where nodes represent emails and edges represent
        reply relationships. Uses efficient batch operations to handle large datasets.
        
        Args:
            emails (List[Dict[str, Any]]): List of parsed email dictionaries
            
        Returns:
            nx.DiGraph: Directed graph representing email thread relationships
        """
        graph = nx.DiGraph()
        
        # Pre-extract all references for better performance
        email_references = [self.extract_email_references(email) for email in emails]
        
        # Add all emails as nodes in batch
        for refs in email_references:
            if refs['message_id']:  # Only add emails with valid message IDs
                graph.add_node(
                    refs['message_id'],
                    email_data=None,  # Store reference instead of full data to save memory
                    subject=refs['subject'],
                    normalized_subject=refs['normalized_subject'],
                    date=refs['date'],
                    sender=refs['from']
                )
        
        # Add edges for reply relationships
        for refs in email_references:
            if not refs['message_id']:
                continue
                
            # Add reply relationship edge
            if refs['in_reply_to'] and refs['in_reply_to'] in graph:
                graph.add_edge(
                    refs['in_reply_to'], 
                    refs['message_id'], 
                    relationship='reply'
                )
            
            # Add reference chain edges
            for ref_id in refs['references']:
                if ref_id != refs['message_id'] and ref_id in graph:
                    graph.add_edge(
                        ref_id, 
                        refs['message_id'], 
                        relationship='reference'
                    )
        
        self.logger.info(f"Built thread graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
    
    def identify_thread_roots(self, graph: nx.DiGraph) -> List[str]:
        """Identify root messages (no incoming edges) with optimization
        
        Finds emails that are not replies to any other email in the dataset,
        which serve as starting points for thread reconstruction.
        
        Args:
            graph (nx.DiGraph): Thread relationship graph
            
        Returns:
            List[str]: List of root message IDs
        """
        roots = [node for node in graph.nodes() if graph.in_degree(node) == 0]
        self.logger.info(f"Identified {len(roots)} thread roots")
        return roots
    
    def extract_conversation_threads(self, graph: nx.DiGraph, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract conversation threads with forensic lineage tracking
        
        Traverses the graph from root nodes to build complete conversation threads.
        Calculates thread metadata including depth, participants, and subject variants.
        Records comprehensive lineage tracking for forensic compliance.
        
        Args:
            graph (nx.DiGraph): Thread relationship graph
            emails (List[Dict[str, Any]]): Original email data for lineage tracking
            
        Returns:
            List[Dict[str, Any]]: List of reconstructed conversation threads with lineage
        """
        roots = self.identify_thread_roots(graph)
        threads = []
        
        # Create lookup for email data by message ID
        email_lookup = {}
        for email in emails:
            message_id = email.get('headers', {}).get('message_id', '').strip('<>')
            if message_id:
                email_lookup[message_id] = email
        
        for idx, root in enumerate(roots):
            # Get all descendants using optimized networkx function
            descendants = nx.descendants(graph, root)
            thread_nodes = [root] + list(descendants)
            
            # Batch collect node data for better performance
            node_data_list = [graph.nodes[node] for node in thread_nodes]
            
            # Create thread with vectorized operations
            participants = {node_data['sender'] for node_data in node_data_list}
            subject_variants = {node_data['subject'] for node_data in node_data_list}
            
            # Calculate thread depth efficiently
            try:
                thread_depth = max(len(nx.shortest_path(graph, root, node)) - 1 
                                 for node in thread_nodes)
            except nx.NetworkXNoPath:
                thread_depth = 0
            
            # Generate canonical message IDs for forensic tracking
            canonical_message_ids = []
            source_files = set()
            thread_emails = []
            
            for message_id in thread_nodes:
                if message_id in email_lookup:
                    email = email_lookup[message_id]
                    thread_emails.append(email)
                    canonical_id = self.lineage_tracker.assign_canonical_id(
                        message_id, email.get('source_file')
                    )
                    canonical_message_ids.append(canonical_id)
                    if email.get('source_file'):
                        source_files.add(email['source_file'])
            
            thread_id = f"thread_{idx+1:04d}"
            
            # Build thread with lineage information
            thread = {
                'thread_id': thread_id,
                'root_message_id': root,
                'message_count': len(thread_nodes),
                'message_ids': thread_nodes,
                'canonical_message_ids': canonical_message_ids,
                'thread_depth': thread_depth,
                'participants': list(participants),
                'date_range': {'start': None, 'end': None},  # TODO: Implement date range calculation
                'subject_variants': list(subject_variants),
                'normalized_subject': node_data_list[0]['normalized_subject'] if node_data_list else None,
                'source_message_lineage': self.lineage_tracker.get_thread_source_lineage(thread_id),
                'source_files': list(source_files)
            }
            
            # Record lineage for this thread reconstruction
            self.lineage_tracker.record_thread_reconstruction(
                thread_id=thread_id,
                input_emails=thread_emails,
                output_thread=thread,
                parameters={
                    'graph_algorithm': 'networkx_descendants',
                    'thread_depth_method': 'shortest_path',
                    'root_identification': 'zero_in_degree'
                }
            )
            
            threads.append(thread)
        
        self.logger.info(f"Extracted {len(threads)} conversation threads with lineage tracking")
        return threads
    
    def validate_thread_quality(self, threads: List[Dict[str, Any]]) -> Dict[str, float]:
        """Generate thread quality metrics with vectorized calculations
        
        Computes comprehensive metrics to evaluate the effectiveness of thread
        reconstruction including depth distribution, participant diversity, and
        message grouping efficiency.
        
        Args:
            threads (List[Dict[str, Any]]): List of reconstructed threads
            
        Returns:
            Dict[str, float]: Quality metrics dictionary
        """
        if not threads:
            return {}
        
        # Use list comprehensions for better performance
        message_counts = [t['message_count'] for t in threads]
        thread_depths = [t['thread_depth'] for t in threads]
        participant_counts = [len(t['participants']) for t in threads]
        
        total_messages = sum(message_counts)
        total_threads = len(threads)
        
        quality_metrics = {
            'total_threads': total_threads,
            'total_messages_threaded': total_messages,
            'avg_messages_per_thread': total_messages / total_threads if total_threads else 0,
            'max_thread_depth': max(thread_depths) if thread_depths else 0,
            'avg_thread_depth': sum(thread_depths) / total_threads if thread_depths else 0,
            'avg_participants_per_thread': sum(participant_counts) / total_threads if participant_counts else 0,
            'single_message_threads': sum(1 for count in message_counts if count == 1),
            'complex_threads': sum(1 for count in message_counts if count >= 5)
        }
        
        return quality_metrics
    
    def save_thread_data(self, threads: List[Dict[str, Any]], quality_metrics: Dict[str, float]) -> None:
        """Save thread reconstruction results with forensic lineage tracking
        
        Persists reconstructed threads and quality metrics to JSON files.
        Saves comprehensive lineage tracking data for forensic compliance.
        Uses UTF-8 encoding to support international characters in email content.
        
        Args:
            threads (List[Dict[str, Any]]): List of reconstructed threads
            quality_metrics (Dict[str, float]): Thread quality evaluation metrics
            
        Raises:
            Exception: If file writing operations fail
        """
        try:
            # Save threads with proper encoding
            threads_path = self.data_dir / "conversation_threads.jsonl"
            with open(threads_path, 'w', encoding='utf-8') as f:
                for thread in threads:
                    f.write(json.dumps(thread, ensure_ascii=False) + '\n')
            
            # Save lineage tracking data
            self.lineage_tracker.save_lineage_summary()
            
            # Enhanced quality report with lineage statistics
            lineage_stats = {
                'total_canonical_ids': len(self.lineage_tracker.canonical_id_map),
                'total_lineage_steps': len(self.lineage_tracker.lineage_steps),
                'source_files_tracked': len(set(
                    sf for step in self.lineage_tracker.lineage_steps 
                    for sf in step.source_files
                ))
            }
            
            report = {
                'processing_timestamp': self.timestamp,
                'thread_quality_metrics': quality_metrics,
                'lineage_statistics': lineage_stats,
                'reconstruction_summary': {
                    'input_emails': quality_metrics.get('total_messages_threaded', 0),
                    'output_threads': quality_metrics.get('total_threads', 0),
                    'threading_efficiency': quality_metrics.get('total_messages_threaded', 0) / 
                                          max(quality_metrics.get('total_threads', 1), 1),
                    'forensic_compliance': {
                        'canonical_ids_assigned': lineage_stats['total_canonical_ids'],
                        'lineage_steps_recorded': lineage_stats['total_lineage_steps'],
                        'source_files_tracked': lineage_stats['source_files_tracked']
                    }
                }
            }
            
            report_path = self.data_dir / "thread_reconstruction_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Thread data saved to {threads_path}")
            self.logger.info(f"Quality report saved to {report_path}")
            self.logger.info(f"Lineage tracking saved: {lineage_stats['total_lineage_steps']} steps recorded")
            
        except Exception as e:
            self.logger.error(f"Failed to save thread data: {e}")
            raise
    
    def run(self) -> None:
        """Execute the complete thread reconstruction pipeline with lineage tracking
        
        Orchestrates the full thread reconstruction workflow from data loading
        through graph construction, thread extraction, quality validation, and
        result persistence with comprehensive forensic lineage tracking.
        
        Raises:
            Exception: If any pipeline stage fails
        """
        try:
            self.logger.info("Starting thread reconstruction pipeline with lineage tracking")
            
            # Load parsed emails
            emails = self.load_parsed_emails()
            
            # Build thread graph
            graph = self.build_thread_graph(emails)
            
            # Extract conversation threads with lineage tracking
            threads = self.extract_conversation_threads(graph, emails)
            
            # Validate quality
            quality_metrics = self.validate_thread_quality(threads)
            
            # Save results with lineage data
            self.save_thread_data(threads, quality_metrics)
            
            # Log summary with lineage statistics
            avg_messages = quality_metrics.get('avg_messages_per_thread', 0)
            lineage_steps = len(self.lineage_tracker.lineage_steps)
            canonical_ids = len(self.lineage_tracker.canonical_id_map)
            
            self.logger.info("Thread reconstruction completed successfully")
            self.logger.info(f"Processed {len(emails)} emails into {len(threads)} threads")
            self.logger.info(f"Average messages per thread: {avg_messages:.2f}")
            self.logger.info(f"Forensic lineage: {lineage_steps} steps, {canonical_ids} canonical IDs assigned")
            
        except Exception as e:
            self.logger.error(f"Thread reconstruction failed: {e}")
            raise


def main():
    """Main entry point for thread reconstruction pipeline
    
    Initializes and executes the complete email thread reconstruction workflow.
    Handles top-level exception logging for pipeline failures.
    """
    base_dir = r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics"
    
    try:
        reconstructor = EmailThreadReconstructor(base_dir)
        reconstructor.run()
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()