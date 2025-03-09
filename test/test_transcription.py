import pytest
import requests
import json
import os
import time
import re
from requests.exceptions import ConnectionError

# Base URL of the transcribe service using the Docker service name
BASE_URL = "http://transcribe:8000/v1/audio/transcriptions"  # Using Docker service name
HEALTHCHECK_URL = "http://transcribe:8000/healthcheck"  # Healthcheck endpoint

# Helper function to load golden data
def load_golden_data(filename):
    filepath = os.path.join(os.path.dirname(__file__), "data", filename)
    with open(filepath, "r") as f:
        return json.load(f)

# Helper function to normalize text for comparison
def normalize_text(text):
    """Normalize text by removing extra whitespace and newlines for comparison."""
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()

# Wait for the transcribe service to be ready
def wait_for_service(max_retries=30, retry_interval=2):
    """Wait for the transcribe service to be ready with retries."""
    print("Waiting for transcribe service to be ready...")
    for attempt in range(max_retries):
        try:
            response = requests.get(HEALTHCHECK_URL)
            if response.status_code == 200:
                print(f"Transcribe service is ready after {attempt+1} attempts!")
                return True
        except ConnectionError:
            print(f"Attempt {attempt+1}/{max_retries}: Service not ready yet, retrying in {retry_interval}s...")
            time.sleep(retry_interval)
    
    print(f"Failed to connect to transcribe service after {max_retries} attempts")
    return False

# Setup function to run before tests
@pytest.fixture(scope="session", autouse=True)
def setup_tests():
    """Setup function that runs once before all tests."""
    assert wait_for_service(), "Transcribe service is not available"

# --- Test Cases ---

def test_transcription_basic():
    """Test basic transcription (no diarization)."""
    audio_file = os.path.join(os.path.dirname(__file__), "data", "single_speaker.m4a")
    golden_data = load_golden_data("single_speaker.json")

    with open(audio_file, "rb") as f:
        response = requests.post(
            BASE_URL,
            files={"file": f},
            data={"model": "medium", "response_format": "verbose_json"},  # Use verbose_json for detailed comparison
        )

    assert response.status_code == 200
    result = response.json()

    # Normalize text for comparison to handle newlines and whitespace differences
    normalized_result = normalize_text(result["text"])
    normalized_golden = normalize_text(golden_data["text"])
    
    # Compare normalized texts
    assert normalized_result == normalized_golden, f"Expected: '{normalized_golden}', Got: '{normalized_result}'"
    
    # Check that segments exist
    assert "segments" in result, "Response should contain segments"

def test_transcription_with_diarization():
    """Test transcription with diarization."""
    audio_file = os.path.join(os.path.dirname(__file__), "data", "multi_speaker.m4a")
    golden_data = load_golden_data("multi_speaker.json")

    with open(audio_file, "rb") as f:
        response = requests.post(
            BASE_URL,
            files={"file": f},
            data={"model": "medium", "diarize": "true", "response_format": "verbose_json"},
        )

    assert response.status_code == 200
    result = response.json()
    
    # Debug: Print the structure of the result
    print("Response structure:")
    print(json.dumps(result, indent=2)[:500] + "...")  # Print first 500 chars
    
    # For diarization, we don't compare the full text directly since the speaker format may vary
    # Instead, we check that the key phrases are present
    result_text = normalize_text(result["text"])
    assert "first speaker" in result_text, "Text should contain 'first speaker'"
    assert "second speaker" in result_text, "Text should contain 'second speaker'"
    
    # Basic checks for diarization
    assert "segments" in result, "Response should contain segments"
    
    # Extract speakers from the response, handling different possible structures
    speakers = set()
    
    # Handle nested segments structure
    segments_data = result["segments"]
    if isinstance(segments_data, dict) and "segments" in segments_data:
        segments_data = segments_data["segments"]
    
    # Process segments
    if isinstance(segments_data, list):
        for segment in segments_data:
            # Check for speaker at segment level
            if isinstance(segment, dict):
                if "speaker" in segment:
                    speakers.add(segment["speaker"])
                # Check for speaker in words
                elif "words" in segment and isinstance(segment["words"], list):
                    for word in segment["words"]:
                        if isinstance(word, dict) and "speaker" in word:
                            speakers.add(word["speaker"])
    
    # Verify we have at least 1 speaker identified
    assert len(speakers) > 0, f"Expected at least 1 speaker, got {len(speakers)}"
    
    # Check number of segments
    if isinstance(segments_data, list):
        assert len(segments_data) > 0, "Should have at least one segment"

# Add more test cases (different languages, edge cases, etc.) as needed. 