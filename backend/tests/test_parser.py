"""
Test CSV Parser

Unit tests for the high-performance CSV parser.
"""

import io
import tempfile
from pathlib import Path

import polars as pl
import pytest

from core.csv_parser import CSVParser


@pytest.fixture
def parser():
    return CSVParser()


@pytest.fixture
def sample_csv_bytes():
    return b"""id,name,value,date
1,Alice,100.5,2024-01-01
2,Bob,200.0,2024-01-02
3,Charlie,150.75,2024-01-03
4,Diana,175.25,2024-01-04
5,Eve,125.0,2024-01-05"""


@pytest.fixture
def sample_csv_file(sample_csv_bytes):
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
        f.write(sample_csv_bytes)
        return Path(f.name)


class TestCSVParser:
    def test_parse_bytes(self, parser, sample_csv_bytes):
        """Test parsing CSV from bytes."""
        df = parser.parse_bytes(sample_csv_bytes)
        
        assert len(df) == 5
        assert len(df.columns) == 4
        assert "id" in df.columns
        assert "name" in df.columns
        assert "value" in df.columns
    
    def test_parse_file(self, parser, sample_csv_file):
        """Test parsing CSV from file."""
        df = parser.parse_file(sample_csv_file)
        
        assert len(df) == 5
        assert len(df.columns) == 4
    
    def test_encoding_detection(self, parser, sample_csv_file):
        """Test encoding detection."""
        encoding = parser.detect_encoding(sample_csv_file)
        
        assert encoding is not None
        assert encoding.lower() in ["utf-8", "ascii", "utf-8-sig"]
    
    def test_session_id_generation(self, parser):
        """Test unique session ID generation."""
        id1 = parser.generate_session_id("test.csv")
        id2 = parser.generate_session_id("test.csv")
        
        # Should be different due to timestamp
        assert id1 != id2
        assert len(id1) == 16
    
    def test_save_and_load(self, parser, sample_csv_bytes, tmp_path):
        """Test saving and loading DataFrame."""
        # Override upload dir for test
        parser.settings.upload_dir = str(tmp_path)
        
        df = parser.parse_bytes(sample_csv_bytes)
        session_id = "test_session"
        
        # Save
        path = parser.save_dataframe(df, session_id)
        assert path.exists()
        
        # Load
        loaded_df = parser.load_dataframe(session_id)
        assert len(loaded_df) == len(df)
        assert loaded_df.columns == df.columns
    
    def test_type_inference(self, parser):
        """Test that data types are correctly inferred."""
        csv_data = b"""int_col,float_col,str_col,date_col
1,1.5,hello,2024-01-01
2,2.5,world,2024-01-02
3,3.5,test,2024-01-03"""
        
        df = parser.parse_bytes(csv_data)
        
        # Polars should infer types correctly
        assert df["int_col"].dtype in (pl.Int64, pl.Int32)
        assert df["float_col"].dtype in (pl.Float64, pl.Float32)
        assert df["str_col"].dtype in (pl.Utf8, pl.String)
    
    def test_handles_nulls(self, parser):
        """Test handling of null values."""
        csv_data = b"""a,b,c
1,hello,
2,,world
,3,test"""
        
        df = parser.parse_bytes(csv_data)
        
        assert df["a"].null_count() == 1
        assert df["b"].null_count() == 1
        assert df["c"].null_count() == 1
    
    def test_handles_special_characters(self, parser):
        """Test handling of special characters."""
        csv_data = b"""name,description
"John, Doe","Description with ""quotes"""
"Jane","Normal text"
"Bob","Text with\nnewline" """
        
        df = parser.parse_bytes(csv_data)
        
        assert len(df) == 3
        assert "John, Doe" in df["name"].to_list()
