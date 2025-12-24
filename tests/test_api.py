import pytest
import sys
import os
import json

# Add backend and ml folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../ml'))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    rv = client.get('/')
    assert rv.status_code == 200

def test_analyze_endpoint(client):
    payload = {
        "emails": [
            {
                "id": 1, 
                "subject": "Urgent Issue", 
                "body": "The server is down! Help!", 
                "timestamp": "2023-10-27T10:00:00"
            }
        ]
    }
    
    rv = client.post('/api/analyze', json=payload)
    assert rv.status_code == 200
    
    data = json.loads(rv.data)
    assert "results" in data
    assert len(data["results"]) == 1
    
    result = data["results"][0]
    assert result["id"] == 1
    assert "emotion" in result
    assert "urgency_score" in result
    assert "summary" in result
    
    # Check if urgency is float
    assert isinstance(result["urgency_score"], float)

def test_analyze_endpoint_empty(client):
    payload = {"emails": []}
    rv = client.post('/api/analyze', json=payload)
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert len(data["results"]) == 0
