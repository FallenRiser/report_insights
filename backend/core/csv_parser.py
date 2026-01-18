"""
High-Performance CSV Parser

Uses Polars for blazing fast CSV parsing (10-100x faster than Pandas).
Supports streaming for large files and automatic encoding detection.
"""

import hashlib
import io
import os
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Optional, Union

import chardet
import polars as pl

from config import get_settings


class CSVParser:
    """High-performance CSV parser using Polars."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def detect_encoding(self, file_path: Union[str, Path]) -> str:
        """Detect file encoding using chardet."""
        with open(file_path, "rb") as f:
            # Read first 100KB for detection
            raw_data = f.read(102400)
        
        result = chardet.detect(raw_data)
        encoding = result.get("encoding", "utf-8")
        
        # Fallback to utf-8 if detection fails
        return encoding or "utf-8"
    
    def detect_encoding_from_bytes(self, data: bytes) -> str:
        """Detect encoding from bytes."""
        # Use first 100KB for detection
        sample = data[:102400]
        result = chardet.detect(sample)
        encoding = result.get("encoding", "utf-8")
        return encoding or "utf-8"
    
    def parse_file(
        self,
        file_path: Union[str, Path],
        encoding: Optional[str] = None,
        infer_schema_length: int = 10000,
        n_rows: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Parse CSV file with automatic type inference.
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding (auto-detected if None)
            infer_schema_length: Number of rows for schema inference
            n_rows: Limit number of rows to read
            
        Returns:
            Polars DataFrame
        """
        if encoding is None:
            encoding = self.detect_encoding(file_path)
        
        # Polars is extremely fast for CSV parsing
        df = pl.read_csv(
            file_path,
            encoding=encoding,
            infer_schema_length=infer_schema_length,
            n_rows=n_rows,
            try_parse_dates=True,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
        
        # Post-process to parse date strings that weren't auto-detected
        df = self._parse_date_strings(df)
        
        return df
    
    def parse_bytes(
        self,
        data: bytes,
        filename: str = "upload.csv",
        infer_schema_length: int = 10000,
    ) -> pl.DataFrame:
        """
        Parse CSV from bytes.
        
        Args:
            data: Raw CSV bytes
            filename: Original filename (for logging)
            infer_schema_length: Number of rows for schema inference
            
        Returns:
            Polars DataFrame
        """
        encoding = self.detect_encoding_from_bytes(data)
        
        # Decode bytes to string
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            # Fallback to latin-1 which accepts any byte
            text = data.decode("latin-1")
        
        # Parse from string
        df = pl.read_csv(
            io.StringIO(text),
            infer_schema_length=infer_schema_length,
            try_parse_dates=True,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
        
        # Post-process to parse date strings that weren't auto-detected
        df = self._parse_date_strings(df)
        
        return df
    
    def _parse_date_strings(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert string columns that look like dates to proper datetime.
        
        This handles common date formats that Polars doesn't auto-detect.
        """
        # Common date patterns in column names
        date_patterns = ["date", "time", "datetime", "timestamp", "created", "updated"]
        
        # Common date formats to try
        date_formats = [
            "%m/%d/%Y",       # 01/15/2024
            "%d/%m/%Y",       # 15/01/2024
            "%Y-%m-%d",       # 2024-01-15
            "%Y/%m/%d",       # 2024/01/15
            "%m-%d-%Y",       # 01-15-2024
            "%d-%m-%Y",       # 15-01-2024
            "%B %d, %Y",      # January 15, 2024
            "%b %d, %Y",      # Jan 15, 2024
            "%d %B %Y",       # 15 January 2024
            "%d %b %Y",       # 15 Jan 2024
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            dtype = df[col].dtype
            
            # Only process string columns with date-like names
            if dtype not in (pl.Utf8, pl.String):
                continue
                
            if not any(pattern in col_lower for pattern in date_patterns):
                continue
            
            # Try each date format
            for fmt in date_formats:
                try:
                    parsed = df[col].str.to_datetime(fmt, strict=False)
                    # Check if parsing succeeded (non-null values)
                    if parsed.null_count() < len(df) * 0.5:
                        df = df.with_columns(parsed.alias(col))
                        break
                except Exception:
                    continue
        
        return df
    
    def parse_streaming(
        self,
        file_path: Union[str, Path],
        chunk_size: Optional[int] = None,
    ) -> pl.LazyFrame:
        """
        Create a lazy frame for streaming large files.
        
        This is memory-efficient for files larger than RAM.
        
        Args:
            file_path: Path to CSV file
            chunk_size: Rows per chunk (uses config default if None)
            
        Returns:
            Polars LazyFrame for lazy evaluation
        """
        if chunk_size is None:
            chunk_size = self.settings.analysis.chunk_size
        
        encoding = self.detect_encoding(file_path)
        
        # LazyFrame for streaming operations
        lazy_df = pl.scan_csv(
            file_path,
            encoding=encoding,
            try_parse_dates=True,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
        
        return lazy_df
    
    def save_dataframe(
        self,
        df: pl.DataFrame,
        session_id: str,
    ) -> Path:
        """
        Save DataFrame to disk for session persistence.
        
        Uses Parquet format for fast loading and compression.
        
        Args:
            df: Polars DataFrame
            session_id: Session identifier
            
        Returns:
            Path to saved file
        """
        upload_dir = Path(self.settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{session_id}.parquet"
        df.write_parquet(file_path, compression="zstd")
        
        return file_path
    
    def load_dataframe(self, session_id: str) -> pl.DataFrame:
        """
        Load DataFrame from session storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Polars DataFrame
        """
        upload_dir = Path(self.settings.upload_dir)
        file_path = upload_dir / f"{session_id}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        
        return pl.read_parquet(file_path)
    
    def load_lazy(self, session_id: str) -> pl.LazyFrame:
        """
        Load DataFrame as LazyFrame for memory efficiency.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Polars LazyFrame
        """
        upload_dir = Path(self.settings.upload_dir)
        file_path = upload_dir / f"{session_id}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        
        return pl.scan_parquet(file_path)
    
    def generate_session_id(self, filename: str) -> str:
        """
        Generate unique session ID based on filename and timestamp.
        
        Args:
            filename: Original filename
            
        Returns:
            Unique session ID
        """
        timestamp = datetime.now().isoformat()
        content = f"{filename}_{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        upload_dir = Path(self.settings.upload_dir)
        file_path = upload_dir / f"{session_id}.parquet"
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False


# Global parser instance
csv_parser = CSVParser()
