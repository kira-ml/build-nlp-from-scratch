import pandas as pd
import numpy as np
import json
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

@dataclass
class ProfileMetrics:
    """Data class for storing profiling metrics"""
    field_name: str
    data_type: str
    completeness_pct: float
    unique_count: int
    unique_ratio: float
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    common_patterns: Optional[List[Tuple[str, int]]] = None
    anomalies: Optional[List[str]] = None

@dataclass 
class DataQualityReport:
    """Comprehensive data quality report"""
    dataset_info: Dict[str, Any]
    field_profiles: List[ProfileMetrics]
    encoding_issues: Dict[str, Any]
    duplicate_analysis: Dict[str, Any]
    content_analysis: Dict[str, Any]
    lineage_info: Optional[Dict[str, Any]] = None
    recommendations: List[str] = None
    
class EmailDataProfiler:
    """Advanced email dataset profiler with forensic-grade analysis"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiling_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    
    def load_email_dataset(self, file_path: str) -> pd.DataFrame:
        """Load and perform initial validation of email dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Loaded dataset with {df.shape[0]} emails and {df.shape[1]} fields")
            
            # Validate expected columns
            expected_cols = ['file', 'message', 'from', 'to', 'subject', 'date']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                warnings.warn(f"Missing expected columns: {missing_cols}")
            
            # Basic data type inference
            print("\n📊 Initial data summary:")
            df.info(memory_usage='deep')
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def analyze_field_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced field completeness analysis with severity classification"""
        completeness_data = []
        
        for col in df.columns:
            null_count = df[col].isnull().sum()
            completeness_pct = ((len(df) - null_count) / len(df)) * 100
            
            # Classify severity
            if completeness_pct >= 95:
                severity = "excellent"
            elif completeness_pct >= 80:
                severity = "good"
            elif completeness_pct >= 50:
                severity = "poor"
            else:
                severity = "critical"
            
            completeness_data.append({
                "field_name": col,
                "completeness_percentage": completeness_pct,
                "missing_count": null_count,
                "severity": severity
            })
        
        completeness_df = pd.DataFrame(completeness_data)
        completeness_df = completeness_df.sort_values("completeness_percentage")
        
        print("📈 Field Completeness Analysis:")
        print(completeness_df.to_string(index=False))
        return completeness_df
    
    def profile_text_fields(self, df: pd.DataFrame, text_columns: List[str]) -> Dict[str, ProfileMetrics]:
        """Advanced profiling of text fields with pattern detection"""
        profiles = {}
        
        for col in text_columns:
            if col not in df.columns:
                continue
                
            text_series = df[col].dropna()
            if len(text_series) == 0:
                continue
            
            # Length statistics
            lengths = text_series.astype(str).str.len()
            
            # Pattern detection for emails, dates, etc.
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
            
            email_matches = text_series.astype(str).str.findall(email_pattern)
            
            # Detect common patterns
            if col == 'from' or col == 'to':
                # Email address analysis
                all_emails = [email for emails in email_matches for email in emails]
                email_domains = [email.split('@')[1] for email in all_emails if '@' in email]
                common_domains = Counter(email_domains).most_common(10)
                patterns = [(f"@{domain}", count) for domain, count in common_domains]
            else:
                # General text patterns
                patterns = []
            
            # Anomaly detection
            anomalies = []
            if lengths.std() > 0:
                z_scores = np.abs((lengths - lengths.mean()) / lengths.std())
                anomaly_indices = text_series.index[z_scores > 3]
                anomalies = [f"Extreme length: idx {idx}" for idx in anomaly_indices[:5]]
            
            profiles[col] = ProfileMetrics(
                field_name=col,
                data_type=str(text_series.dtype),
                completeness_pct=((len(text_series) / len(df)) * 100),
                unique_count=text_series.nunique(),
                unique_ratio=text_series.nunique() / len(text_series),
                min_length=int(lengths.min()),
                max_length=int(lengths.max()),
                avg_length=float(lengths.mean()),
                common_patterns=patterns,
                anomalies=anomalies
            )
        
        return profiles
    
    def detect_encoding_issues(self, df: pd.DataFrame, text_columns: List[str]) -> Dict[str, Any]:
        """Advanced encoding issue detection with categorization"""
    def detect_encoding_issues(self, df: pd.DataFrame, text_columns: List[str]) -> Dict[str, Any]:
        """Advanced encoding issue detection with categorization"""
        encoding_analysis = {
            "total_issues": 0,
            "issues_by_column": {},
            "issue_types": {
                "unicode_decode_error": 0,
                "unicode_encode_error": 0,
                "suspicious_chars": 0,
                "non_printable": 0
            },
            "problematic_records": []
        }
        
        for col in text_columns:
            if col not in df.columns:
                continue
                
            column_issues = []
            
            for idx, text in df[col].items():
                if pd.isna(text):
                    continue
                    
                text_str = str(text)
                issue_types = []
                
                # Test UTF-8 encoding/decoding
                try:
                    text_str.encode("utf-8").decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError) as e:
                    issue_types.append("unicode_error")
                    encoding_analysis["issue_types"]["unicode_decode_error"] += 1
                
                # Check for suspicious characters
                suspicious_chars = re.findall(r'[^\x00-\x7F\x80-\xFF]', text_str)
                if suspicious_chars:
                    issue_types.append("suspicious_chars")
                    encoding_analysis["issue_types"]["suspicious_chars"] += 1
                
                # Check for non-printable characters
                non_printable = re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', text_str)
                if non_printable:
                    issue_types.append("non_printable")
                    encoding_analysis["issue_types"]["non_printable"] += 1
                
                if issue_types:
                    column_issues.append({
                        "index": idx,
                        "issues": issue_types,
                        "text_preview": text_str[:100] + "..." if len(text_str) > 100 else text_str
                    })
            
            encoding_analysis["issues_by_column"][col] = {
                "count": len(column_issues),
                "issues": column_issues[:10]  # Limit to first 10 for reporting
            }
            encoding_analysis["total_issues"] += len(column_issues)
        
        print(f"🔍 Encoding Analysis: Found {encoding_analysis['total_issues']} total issues")
        for col, data in encoding_analysis["issues_by_column"].items():
            if data["count"] > 0:
                print(f"  └─ {col}: {data['count']} issues")
        
        return encoding_analysis
    
    def identify_duplicates(self, df: pd.DataFrame, subset_fields: List[str] = None) -> Dict[str, Any]:
        """Enhanced duplicate detection with multiple strategies"""
        if subset_fields is None:
            subset_fields = ["file", "message"]
        
        # Ensure subset fields exist
        available_fields = [f for f in subset_fields if f in df.columns]
        if not available_fields:
            print("⚠️ No valid fields for duplicate detection")
            return {"duplicate_count": 0, "strategies": {}}
        
        duplicate_analysis = {
            "strategies": {},
            "duplicate_count": 0,
            "duplicate_indices": set()
        }
        
        # Strategy 1: Exact field matching
        exact_duplicates = df[df.duplicated(subset=available_fields, keep=False)]
        duplicate_analysis["strategies"]["exact_match"] = {
            "count": len(exact_duplicates),
            "method": f"Exact match on {available_fields}",
            "indices": exact_duplicates.index.tolist()
        }
        duplicate_analysis["duplicate_indices"].update(exact_duplicates.index)
        
        # Strategy 2: Content hash matching (if message column exists)
        if "message" in df.columns:
            df_clean = df.dropna(subset=["message"])
            content_hashes = df_clean["message"].apply(
                lambda x: hashlib.md5(str(x).encode()).hexdigest()
            )
            hash_duplicates = df_clean[content_hashes.duplicated(keep=False)]
            duplicate_analysis["strategies"]["content_hash"] = {
                "count": len(hash_duplicates),
                "method": "MD5 hash of message content",
                "indices": hash_duplicates.index.tolist()
            }
            duplicate_analysis["duplicate_indices"].update(hash_duplicates.index)
        
        # Strategy 3: Fuzzy matching for near-duplicates (simplified)
        if "subject" in df.columns:
            subject_normalized = df["subject"].fillna("").str.lower().str.strip()
            subject_duplicates = df[subject_normalized.duplicated(keep=False)]
            duplicate_analysis["strategies"]["subject_fuzzy"] = {
                "count": len(subject_duplicates),
                "method": "Normalized subject matching",
                "indices": subject_duplicates.index.tolist()
            }
        
        duplicate_analysis["duplicate_count"] = len(duplicate_analysis["duplicate_indices"])
        
        print(f"🔍 Duplicate Analysis:")
        for strategy, data in duplicate_analysis["strategies"].items():
            print(f"  └─ {strategy}: {data['count']} duplicates ({data['method']})")
        
        return duplicate_analysis
    
    def analyze_content_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content patterns for forensic insights"""
        content_analysis = {
            "email_volume_by_sender": {},
            "timestamp_patterns": {},
            "content_statistics": {},
            "communication_networks": {}
        }
        
        # Email volume analysis
        if "from" in df.columns:
            sender_counts = df["from"].value_counts().head(20)
            content_analysis["email_volume_by_sender"] = sender_counts.to_dict()
        
        # Content length analysis
        if "message" in df.columns:
            message_lengths = df["message"].fillna("").astype(str).str.len()
            content_analysis["content_statistics"] = {
                "avg_message_length": float(message_lengths.mean()),
                "median_message_length": float(message_lengths.median()),
                "max_message_length": int(message_lengths.max()),
                "messages_over_1000_chars": int((message_lengths > 1000).sum()),
                "empty_messages": int((message_lengths == 0).sum())
            }
        
        # Time-based patterns (if date column exists and is parseable)
        if "date" in df.columns:
            try:
                dates = pd.to_datetime(df["date"], errors='coerce')
                valid_dates = dates.dropna()
                if len(valid_dates) > 0:
                    content_analysis["timestamp_patterns"] = {
                        "date_range": {
                            "earliest": valid_dates.min().isoformat(),
                            "latest": valid_dates.max().isoformat()
                        },
                        "emails_by_hour": valid_dates.dt.hour.value_counts().head(10).to_dict(),
                        "emails_by_weekday": valid_dates.dt.day_name().value_counts().to_dict()
                    }
            except Exception as e:
                print(f"⚠️ Could not parse dates: {e}")
        
        return content_analysis
    
    def create_visualizations(self, df: pd.DataFrame, profiles: Dict[str, ProfileMetrics], 
                            output_dir: Path) -> List[str]:
        """Generate visualization reports"""
        viz_files = []
        
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Field completeness visualization
        if profiles:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Completeness bar chart
            field_names = [p.field_name for p in profiles.values()]
            completeness_pcts = [p.completeness_pct for p in profiles.values()]
            
            bars = ax1.bar(field_names, completeness_pcts)
            ax1.set_title("Field Completeness Analysis")
            ax1.set_ylabel("Completeness Percentage")
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='x', rotation=45)
            
            # Color code bars
            for bar, pct in zip(bars, completeness_pcts):
                if pct >= 95:
                    bar.set_color('green')
                elif pct >= 80:
                    bar.set_color('orange') 
                else:
                    bar.set_color('red')
            
            # Text length distribution
            if 'message' in profiles:
                ax2.hist([profiles['message'].min_length, profiles['message'].avg_length, 
                         profiles['message'].max_length], bins=20, alpha=0.7)
                ax2.set_title("Message Length Distribution")
                ax2.set_xlabel("Character Count")
                ax2.set_ylabel("Frequency")
            
            plt.tight_layout()
            viz_file = output_dir / f"data_quality_overview_{self.profiling_timestamp}.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files.append(str(viz_file))
        
        return viz_files
    
    def generate_comprehensive_report(self, df: pd.DataFrame, 
                                    lineage_file: Optional[str] = None) -> DataQualityReport:
        """Generate comprehensive data quality report with lineage integration"""
        
        print("🔍 Starting comprehensive data profiling...")
        
        # Basic dataset info
        dataset_info = {
            "total_records": len(df),
            "total_fields": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "profiling_timestamp": self.profiling_timestamp,
            "columns": df.columns.tolist()
        }
        
        # Field completeness analysis
        completeness_df = self.analyze_field_completeness(df)
        
        # Text field profiling
        text_columns = ["message", "subject", "from", "to"]
        field_profiles = list(self.profile_text_fields(df, text_columns).values())
        
        # Encoding analysis
        encoding_issues = self.detect_encoding_issues(df, text_columns)
        
        # Duplicate analysis
        duplicate_analysis = self.identify_duplicates(df)
        
        # Content pattern analysis
        content_analysis = self.analyze_content_patterns(df)
        
        # Lineage integration
        lineage_info = None
        if lineage_file and Path(lineage_file).exists():
            try:
                with open(lineage_file, 'r') as f:
                    lineage_info = json.load(f)
                print("✓ Integrated lineage information")
            except Exception as e:
                print(f"⚠️ Could not load lineage file: {e}")
        
        # Generate visualizations
        viz_files = self.create_visualizations(df, 
                                             {p.field_name: p for p in field_profiles}, 
                                             self.output_dir)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            completeness_df, encoding_issues, duplicate_analysis, content_analysis
        )
        
        # Create comprehensive report
        report = DataQualityReport(
            dataset_info=dataset_info,
            field_profiles=field_profiles,
            encoding_issues=encoding_issues,
            duplicate_analysis=duplicate_analysis,
            content_analysis=content_analysis,
            lineage_info=lineage_info,
            recommendations=recommendations
        )
        
        # Save report
        report_file = self.output_dir / f"comprehensive_quality_report_{self.profiling_timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            # Convert dataclass to dict for JSON serialization
            report_dict = asdict(report)
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"✅ Comprehensive report saved to: {report_file}")
        print(f"📊 Visualizations saved: {len(viz_files)} files")
        
        return report
    
    def _generate_recommendations(self, completeness_df: pd.DataFrame, 
                                encoding_issues: Dict, duplicate_analysis: Dict,
                                content_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Completeness recommendations
        poor_fields = completeness_df[completeness_df['completeness_percentage'] < 80]
        if len(poor_fields) > 0:
            recommendations.append(
                f"🔧 Address poor completeness in fields: {', '.join(poor_fields['field_name'].tolist())}"
            )
        
        # Encoding recommendations
        if encoding_issues['total_issues'] > 0:
            recommendations.append(
                f"🔤 Fix {encoding_issues['total_issues']} encoding issues before NLP processing"
            )
        
        # Duplicate recommendations
        if duplicate_analysis['duplicate_count'] > 0:
            recommendations.append(
                f"🗂️ Remove or mark {duplicate_analysis['duplicate_count']} duplicates for data integrity"
            )
        
        # Content-specific recommendations
        if 'content_statistics' in content_analysis:
            stats = content_analysis['content_statistics']
            if stats['empty_messages'] > 0:
                recommendations.append(
                    f"📝 Investigate {stats['empty_messages']} empty messages"
                )
        
        # Forensic recommendations
        recommendations.extend([
            "🔍 Implement content hash verification for forensic integrity",
            "📊 Consider thread reconstruction for temporal analysis", 
            "🔗 Establish lineage tracking for all transformations",
            "⚡ Use sampling for large datasets to optimize processing"
        ])
        
        return recommendations

# Example usage and backwards compatibility
def run_profiling_analysis(dataset_path: str, lineage_file: Optional[str] = None) -> DataQualityReport:
    """Main function to run comprehensive profiling analysis"""
    profiler = EmailDataProfiler()
    df = profiler.load_email_dataset(dataset_path)
    report = profiler.generate_comprehensive_report(df, lineage_file)
    return report

# Backwards compatibility functions
def load_email_dataset(file_path: str) -> pd.DataFrame:
    """Legacy function for backwards compatibility"""
    profiler = EmailDataProfiler()
    return profiler.load_email_dataset(file_path)

def analyze_field_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy function for backwards compatibility"""
    profiler = EmailDataProfiler()
    return profiler.analyze_field_completeness(df)

if __name__ == "__main__":
    # Example usage
    dataset_path = r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics\data\processed\emails_sampled_5k.csv"
    
    # Check if lineage file exists from content processor
    lineage_file = r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics\data\processed\transformation_lineage_summary.json"
    
    try:
        print("🚀 Starting optimized email data profiling...")
        report = run_profiling_analysis(dataset_path, lineage_file)
        print("✅ Profiling complete! Check the reports/ directory for detailed analysis.")
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        # Fallback to basic analysis
        profiler = EmailDataProfiler()
        df = profiler.load_email_dataset(dataset_path)
        basic_completeness = profiler.analyze_field_completeness(df)
        print("📊 Basic analysis completed")