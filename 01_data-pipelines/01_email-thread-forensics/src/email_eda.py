"""
Email Dataset Exploratory Data Analysis (EDA)
NLP-focused analysis for email forensics and communication patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import email
from email.utils import parsedate_tz
from datetime import datetime
import os
import time
import calendar

# Set matplotlib style for dark mode visualizations
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['figure.facecolor'] = '#1e1e1e'
plt.rcParams['axes.facecolor'] = '#2d2d2d'

def load_and_inspect_dataset(file_path):
    """
    Load email dataset and perform initial inspection
    """
    print("="*60)
    print("EMAIL DATASET INSPECTION")
    print("="*60)
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Names: {list(df.columns)}")
    
    # Basic info
    print(f"\nDataset Info:")
    print(df.info())
    
    # Missing values
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")
    
    # Sample data
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    return df

def preprocess_text(text):
    """
    Basic text preprocessing for NLP analysis
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove email headers and metadata
    text = re.sub(r'message-id:.*?\n', '', text)
    text = re.sub(r'from:.*?\n', '', text)
    text = re.sub(r'to:.*?\n', '', text)
    text = re.sub(r'subject:.*?\n', '', text)
    text = re.sub(r'date:.*?\n', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_email_length_stats(df, text_column):
    """
    Calculate email length statistics (characters and estimated tokens)
    """
    print("\n" + "="*60)
    print("EMAIL LENGTH ANALYSIS")
    print("="*60)
    
    # Character length
    char_lengths = df[text_column].astype(str).str.len()
    
    # Estimated token count (split by whitespace)
    token_lengths = df[text_column].astype(str).apply(lambda x: len(x.split()))
    
    print(f"\nCharacter Length Statistics:")
    print(f"Mean: {char_lengths.mean():.1f}")
    print(f"Median: {char_lengths.median():.1f}")
    print(f"Min: {char_lengths.min()}")
    print(f"Max: {char_lengths.max()}")
    print(f"Std: {char_lengths.std():.1f}")
    
    print(f"\nToken Length Statistics:")
    print(f"Mean: {token_lengths.mean():.1f}")
    print(f"Median: {token_lengths.median():.1f}")
    print(f"Min: {token_lengths.min()}")
    print(f"Max: {token_lengths.max()}")
    print(f"Std: {token_lengths.std():.1f}")
    
    return char_lengths, token_lengths

def plot_length_distributions(char_lengths, token_lengths):
    """
    Plot email length distributions
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Character length distribution
    ax1.hist(char_lengths, bins=50, alpha=0.7, color='#2196F3', edgecolor='white')
    ax1.set_title('Distribution of Email Lengths (Characters)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Characters')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Token length distribution
    ax2.hist(token_lengths, bins=50, alpha=0.7, color='#FF5722', edgecolor='white')
    ax2.set_title('Distribution of Email Lengths (Tokens)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Tokens')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('email_length_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_word_frequency(df, text_column, top_n=20):
    """
    Analyze word frequency in emails after preprocessing
    """
    print("\n" + "="*60)
    print("WORD FREQUENCY ANALYSIS")
    print("="*60)
    
    # Common English stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
                'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
                'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
                'can', 'will', 'just', 'don', 'should', 'now', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 
                'them', 'their', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 
                'get', 'got', 'make', 'made', 'go', 'went', 'come', 'came'}
    
    # Combine all text and preprocess
    all_text = ' '.join(df[text_column].astype(str))
    processed_text = preprocess_text(all_text)
    
    # Tokenize and filter
    words = processed_text.split()
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Count frequency
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(top_n)
    
    print(f"\nTop {top_n} Most Frequent Words:")
    for i, (word, count) in enumerate(top_words, 1):
        print(f"{i:2d}. {word:<15} ({count:,} occurrences)")
    
    return top_words

def plot_word_frequency(top_words, top_n=20):
    """
    Plot word frequency bar chart
    """
    words, counts = zip(*top_words[:top_n])
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(words)), counts, color='#4CAF50', edgecolor='white', alpha=0.8)
    
    plt.title(f'Top {top_n} Most Frequent Words in Emails', fontsize=16, fontweight='bold')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('word_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def extract_email_metadata(df, message_column):
    """
    Extract sender, recipient, and other metadata from email messages
    """
    print("\n" + "="*60)
    print("EMAIL METADATA EXTRACTION")
    print("="*60)
    
    senders = []
    recipients = []
    subjects = []
    dates = []
    
    for idx, message in df[message_column].items():
        try:
            # Parse email message
            email_obj = email.message_from_string(str(message))
            
            # Extract metadata
            sender = email_obj.get('From', 'Unknown')
            recipient = email_obj.get('To', 'Unknown')
            subject = email_obj.get('Subject', 'No Subject')
            date_str = email_obj.get('Date', '')
            
            senders.append(sender)
            recipients.append(recipient)
            subjects.append(subject)
            dates.append(date_str)
            
        except Exception as e:
            senders.append('Unknown')
            recipients.append('Unknown')
            subjects.append('No Subject')
            dates.append('')
    
    # Add to dataframe
    df['sender'] = senders
    df['recipient'] = recipients
    df['subject'] = subjects
    df['date_str'] = dates
    
    return df

def analyze_communication_patterns(df):
    """
    Analyze sender/recipient activity patterns
    """
    print("\n" + "="*60)
    print("COMMUNICATION PATTERNS")
    print("="*60)
    
    # Top senders
    print("\nTop 10 Most Active Senders:")
    sender_counts = df['sender'].value_counts().head(10)
    for i, (sender, count) in enumerate(sender_counts.items(), 1):
        print(f"{i:2d}. {sender:<40} ({count} emails)")
    
    # Top recipients
    print("\nTop 10 Most Active Recipients:")
    recipient_counts = df['recipient'].value_counts().head(10)
    for i, (recipient, count) in enumerate(recipient_counts.items(), 1):
        print(f"{i:2d}. {recipient:<40} ({count} emails)")
    
    return sender_counts, recipient_counts

def plot_communication_activity(sender_counts, recipient_counts):
    """
    Plot sender and recipient activity
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Top senders
    senders = sender_counts.head(10).index
    sender_values = sender_counts.head(10).values
    
    bars1 = ax1.barh(range(len(senders)), sender_values, color='#03A9F4', edgecolor='white')
    ax1.set_title('Top 10 Most Active Email Senders', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Emails Sent')
    ax1.set_ylabel('Senders')
    ax1.set_yticks(range(len(senders)))
    ax1.set_yticklabels([s[:30] + '...' if len(s) > 30 else s for s in senders])
    
    # Add value labels
    for bar, value in zip(bars1, sender_values):
        ax1.text(bar.get_width() + max(sender_values)*0.01, bar.get_y() + bar.get_height()/2, 
                str(value), ha='left', va='center', fontweight='bold')
    
    # Top recipients
    recipients = recipient_counts.head(10).index
    recipient_values = recipient_counts.head(10).values
    
    bars2 = ax2.barh(range(len(recipients)), recipient_values, color='#FF5722', edgecolor='white')
    ax2.set_title('Top 10 Most Active Email Recipients', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Emails Received')
    ax2.set_ylabel('Recipients')
    ax2.set_yticks(range(len(recipients)))
    ax2.set_yticklabels([r[:30] + '...' if len(r) > 30 else r for r in recipients])
    
    # Add value labels
    for bar, value in zip(bars2, recipient_values):
        ax2.text(bar.get_width() + max(recipient_values)*0.01, bar.get_y() + bar.get_height()/2, 
                str(value), ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('communication_activity.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_temporal_patterns(df):
    """
    Analyze temporal patterns in email communication
    """
    print("\n" + "="*60)
    print("TEMPORAL ANALYSIS")
    print("="*60)
    
    parsed_dates = []
    
    for date_str in df['date_str']:
        try:
            if date_str and date_str != 'Unknown':
                # Parse email date
                date_tuple = parsedate_tz(date_str)
                if date_tuple:
                    # Convert to timestamp handling timezone
                    if date_tuple[9] is not None:
                        # Has timezone info
                        timestamp = calendar.timegm(date_tuple[:9]) - date_tuple[9]
                    else:
                        # No timezone info, use local time
                        timestamp = time.mktime(date_tuple[:9])
                    dt = datetime.fromtimestamp(timestamp)
                    parsed_dates.append(dt)
                else:
                    parsed_dates.append(None)
            else:
                parsed_dates.append(None)
        except:
            parsed_dates.append(None)
    
    df['parsed_date'] = parsed_dates
    valid_dates = df.dropna(subset=['parsed_date'])
    
    print(f"\nSuccessfully parsed {len(valid_dates)} out of {len(df)} email dates")
    
    if len(valid_dates) > 0:
        # Extract time components
        valid_dates = valid_dates.copy()
        valid_dates['hour'] = valid_dates['parsed_date'].dt.hour
        valid_dates['weekday'] = valid_dates['parsed_date'].dt.day_name()
        valid_dates['date_only'] = valid_dates['parsed_date'].dt.date
        
        return valid_dates
    else:
        print("No valid dates found for temporal analysis")
        return None

def plot_temporal_patterns(valid_dates):
    """
    Plot temporal patterns
    """
    if valid_dates is None or len(valid_dates) == 0:
        print("No temporal data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Hour distribution
    hour_counts = valid_dates['hour'].value_counts().sort_index()
    ax1.bar(hour_counts.index, hour_counts.values, color='#FFC107', edgecolor='white', alpha=0.8)
    ax1.set_title('Email Activity by Hour of Day', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hour (24-hour format)')
    ax1.set_ylabel('Number of Emails')
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, alpha=0.3)
    
    # Weekday distribution
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = valid_dates['weekday'].value_counts().reindex(weekday_order, fill_value=0)
    
    bars = ax2.bar(range(len(weekday_order)), weekday_counts.values, 
                   color='#9C27B0', edgecolor='white', alpha=0.8)
    ax2.set_title('Email Activity by Day of Week', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Number of Emails')
    ax2.set_xticks(range(len(weekday_order)))
    ax2.set_xticklabels(weekday_order, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, weekday_counts.values):
        if value > 0:  # Only add label if there's a value
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(weekday_counts.values)*0.01, 
                    str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(df, char_lengths, token_lengths, top_words, sender_counts, recipient_counts, valid_dates):
    """
    Create a comprehensive summary report
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE EDA SUMMARY REPORT")
    print("="*60)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total emails analyzed: {len(df):,}")
    print(f"   • Data completeness: {((len(df) - df.isnull().sum().sum()) / (len(df) * len(df.columns)) * 100):.1f}%")
    print(f"   • Unique senders: {df['sender'].nunique():,}")
    print(f"   • Unique recipients: {df['recipient'].nunique():,}")
    
    print(f"\n📝 TEXT CHARACTERISTICS:")
    print(f"   • Average email length: {char_lengths.mean():.0f} characters")
    print(f"   • Average tokens per email: {token_lengths.mean():.0f} words")
    print(f"   • Longest email: {char_lengths.max():,} characters")
    print(f"   • Most common word: '{top_words[0][0]}' ({top_words[0][1]:,} occurrences)")
    
    print(f"\n📈 COMMUNICATION PATTERNS:")
    most_active_sender = sender_counts.index[0] if len(sender_counts) > 0 else "Unknown"
    most_active_recipient = recipient_counts.index[0] if len(recipient_counts) > 0 else "Unknown"
    print(f"   • Most active sender: {most_active_sender[:50]}")
    print(f"   • Most active recipient: {most_active_recipient[:50]}")
    print(f"   • Email distribution: Top 10 senders account for {sender_counts.head(10).sum():,} emails")
    
    if valid_dates is not None and len(valid_dates) > 0:
        print(f"\n⏰ TEMPORAL INSIGHTS:")
        peak_hour = valid_dates['hour'].mode().iloc[0] if len(valid_dates['hour'].mode()) > 0 else "Unknown"
        peak_day = valid_dates['weekday'].mode().iloc[0] if len(valid_dates['weekday'].mode()) > 0 else "Unknown"
        print(f"   • Peak activity hour: {peak_hour}:00")
        print(f"   • Most active day: {peak_day}")
        print(f"   • Date range: {valid_dates['parsed_date'].min().strftime('%Y-%m-%d')} to {valid_dates['parsed_date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\n📁 GENERATED FILES:")
    print(f"   • email_length_distributions.png")
    print(f"   • word_frequency_analysis.png")
    print(f"   • communication_activity.png")
    print(f"   • temporal_patterns.png")

def main():
    """
    Main function to execute the complete EDA pipeline
    """
    # File path
    file_path = r"C:\Users\Ken Ira Talingting\Desktop\build-nlp-from-scratch\01_data-pipelines\01_email-thread-forensics\data\processed\emails_sampled_5k.csv"
    
    try:
        # 1. Load and inspect dataset
        df = load_and_inspect_dataset(file_path)
        
        # Determine text column (adjust based on your dataset structure)
        text_column = 'message' if 'message' in df.columns else df.columns[-1]
        
        # 2. Email length analysis
        char_lengths, token_lengths = get_email_length_stats(df, text_column)
        plot_length_distributions(char_lengths, token_lengths)
        
        # 3. Word frequency analysis
        top_words = analyze_word_frequency(df, text_column)
        plot_word_frequency(top_words)
        
        # 4. Extract email metadata
        df = extract_email_metadata(df, text_column)
        
        # 5. Communication patterns
        sender_counts, recipient_counts = analyze_communication_patterns(df)
        plot_communication_activity(sender_counts, recipient_counts)
        
        # 6. Temporal analysis
        valid_dates = analyze_temporal_patterns(df)
        plot_temporal_patterns(valid_dates)
        
        # 7. Generate comprehensive summary
        create_summary_report(df, char_lengths, token_lengths, top_words, sender_counts, recipient_counts, valid_dates)
        
        print("\n" + "="*60)
        print("✅ EDA COMPLETE - All visualizations saved as PNG files")
        print("="*60)
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error during EDA execution: {str(e)}")
        print("Please check your file path and data format.")
        return None

if __name__ == "__main__":
    df_analyzed = main()
