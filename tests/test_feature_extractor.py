import pytest
import numpy as np
import sys
import os

# Add ml folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../ml'))

from feature_extractor import FeatureExtractor

def test_feature_extractor_basic():
    extractor = FeatureExtractor()
    text = "Hello! This is a test."
    features = extractor.extract(text)
    
    # Check shape
    assert features.shape == (6,)
    
    # Check values
    # Exclamations: 1
    assert features[0] == 1.0
    # Questions: 0
    assert features[1] == 0.0
    # Length: 22
    assert features[3] == 22.0

def test_feature_extractor_uppercase():
    extractor = FeatureExtractor()
    text = "HELLO WORLD"
    features = extractor.extract(text)
    
    # Caps ratio should be high (approx 1.0 minus spaces)
    # Caps: 10, Total: 11. Ratio: 10/11 = 0.909
    assert features[2] > 0.9

def test_feature_extractor_empty():
    extractor = FeatureExtractor()
    text = ""
    features = extractor.extract(text)
    assert features.shape == (6,)
    assert features[2] == 0.0 # Caps ratio
    assert features[3] == 0.0 # Length
