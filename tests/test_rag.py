# tests/test_rag.py
import pytest
import pandas as pd
from src.data_loader import ComplaintDataProcessor
from src.risk_analyzer import BusinessRiskAnalyzer

def test_text_cleaning_logic():
    # Arrange
    processor = ComplaintDataProcessor(filepath="fake_path.csv")
    dirty_text = "The bank charged me   unauthorized fees XX/XX/2026 XXXX"
    
    # Act
    cleaned = processor.clean_text(dirty_text)
    
    # Assert
    assert "xx" not in cleaned
    assert "  " not in cleaned  # Double spaces should be flattened
    assert cleaned == "the bank charged me unauthorized fees / /2026"

def test_risk_quantification_high_risk():
    # Arrange
    high_risk_complaint = {
        "cleaned_Consumer complaint narrative": "I will contact my lawyer to file a lawsuit in court for fraud.",
        "Consumer disputed?": "Yes"
    }
    
    # Act
    analysis = BusinessRiskAnalyzer.calculate_risk_score(high_risk_complaint)
    
    # Assert
    assert analysis["risk_score"] >= 40
    assert "CRITICAL" in analysis["severity"]