## Overview

WhisperX API Server is a FastAPI-based server designed to transcribe audio files using the Whisper ASR (Automatic Speech Recognition) model based on WhisperX (https://github.com/m-bain/WhisperX) Python library. The API offers an OpenAI-like interface that allows users to upload audio files and receive transcription results in various formats. It supports customizable options such as different models, languages, temperature settings, and more.

Features
1. Audio Transcription: Transcribe audio files using the Whisper ASR model.
2. Model Caching: Load and cache models for reusability and faster performance.
3. OpenAI-like API, based on https://platform.openai.com/docs/api-reference/audio/createTranscription and https://platform.openai.com/docs/api-reference/audio/createTranslation

## API Endpoints

### `POST /v1/audio/transcriptions`
https://platform.openai.com/docs/api-reference/audio/createTranscription

**Parameters**:
- `file`: The audio file to transcribe.
- `model (str)`: The Whisper model to use. Default is `config.whisper.model`.
- `language (str)`: The language for transcription. Default is `config.default_language`.
- `prompt (str)`: Optional transcription prompt.
- `response_format (str)`: The format of the transcription output. Defaults to `json`.
- `temperature (float)`: Temperature setting for transcription. Default is `0.0`.
- `timestamp_granularities (list)`: Granularity of timestamps, either `segment` or `word`. Default is `["segment"]`.
- `stream (bool)`: Enable streaming mode for real-time transcription. (Doesn't work.)
- `hotwords (str)`: Optional hotwords for transcription.
- `suppress_numerals (bool)`: Option to suppress numerals in the transcription. Default is `True`.
- `highlight_words (bool)`: Highlight words in the transcription output for formats like VTT and SRT.
- `align (bool)`: Option to do transcription timings alignment. Default is `True`.
- `diarize (bool)`: Option to diarize the transcription. Default is `False`.

**Returns**: Transcription results in the specified format.

### `POST /v1/audio/translations`
https://platform.openai.com/docs/api-reference/audio/createTranslation

**Parameters**:
- `file`: The audio file to translate.
- `model (str)`: The Whisper model to use. Default is `config.whisper.model`.
- `prompt (str)`: Optional translation prompt.
- `response_format (str)`: The format of the translation output. Defaults to `json`.
- `temperature (float)`: Temperature setting for translation. Default is `0.0`.

**Returns**: Translation results in the specified format.

### `GET /healthcheck`
Returns the current health status of the API server.

### `GET /models/list`
Lists all loaded models currently available on the server.

### `POST /models/unload`
Unloads a specific model from memory cache.

### `POST /models/load`
Loads a specified model into memory.

### Running the API

**With Docker**:

For CPU:
```bash
    docker compose build whisperx-api-server-cpu

    docker compose up whisperx-api-server-cpu
```

For CUDA (GPU):
```bash
    docker compose build whisperx-api-server-cuda

    docker compose up whisperx-api-server-cuda

```

## Contributing

Feel free to submit issues, fork the repository, and send pull requests to contribute to the project.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3. See the `LICENSE` file for details.

# WhisperX API Server

This project provides a FastAPI-based API server for performing transcription and diarization using the WhisperX library. It's designed to be deployed as a Docker container and includes features for persistent model storage and integration testing.

## 1. Requirements

This document outlines the design for extending the existing WhisperX API server with the following:

*   **New "transcribe" service (CPU-only):**  A new Docker service named `transcribe` will be added to the `compose.yaml` file. This service will utilize the CPU-based WhisperX API server, making transcription and diarization accessible without requiring a GPU.
*   **Persistent Model Storage:** The WhisperX model will be stored in a local `data/models` directory within the project. This prevents repeated downloads of the model each time the service starts.  The `data` directory should be at the project root.
*   **API Functionality:** The API must support both transcription and diarization functionalities through the existing `/v1/audio/transcriptions` endpoint.
*   **Testing Docker Image:** A separate Docker image and associated files will be created to execute tests, verifying the correctness of the transcription and diarization outputs. This ensures the API behaves as expected.

## 2. Implementation Design

### 2.1. `transcribe` Service (docker-compose.yaml)

The `docker-compose.yaml` file (at the project root) will be modified to include a new `transcribe` service. This service will be based on the `whisperx-api-server-cpu` image. Key configurations include:

*   **Image:** `whisperx-api-server-cpu` (This assumes the existing CPU image is suitable. If not, a new `Dockerfile.cpu-transcribe` might be needed, but the current one seems appropriate.)
*   **Build Context:**  The build context will remain the project root (`.`).
*   **Dockerfile:** `Dockerfile.cpu` (again, assuming the existing one is sufficient).
*   **Healthcheck:** The existing healthcheck will be reused.
*   **Command:** The same `uvicorn` command will be used to start the FastAPI application.
*   **Ports:**  Port `8000` will be exposed, consistent with the existing services.
*   **Volumes:**
    *   `./data/models:/workspace/models`: This crucial addition mounts the local `data/models` directory to `/workspace/models` inside the container.  This is where the WhisperX model will be stored persistently.  The application will be configured to use this path.
    *   `hugging_face_cache:/root/.cache/huggingface`:  This existing volume mount will be kept for caching Hugging Face resources.
    *   `torch_cache:/root/.cache/torch`: This existing volume mount will be kept for caching PyTorch resources.
* **Environment Variables**:
    * `WHISPER__DOWNLOAD_ROOT`: Set to `/workspace/models` to instruct WhisperX to download and load models from this persistent location.
* **Container Name**: Set to `transcribe` to ensure the test container can reliably connect to it by name.

### 2.2. Model Loading and Configuration

*   **`config.py`:** The `WhisperConfig` class within `src/whisperx_api_server/config.py` will be updated. The `download_root` field will be set to `/workspace/models` by default. This ensures that the application looks for models in the mounted volume.
* **`main.py`**: No changes are needed here. The lifespan context manager already handles model loading.
*   **`models.py`:** The `initialize_model` function will use the `download_root` from the configuration, ensuring models are loaded from the correct location.

### 2.3. API Functionality (Transcription & Diarization)

The existing `/v1/audio/transcriptions` endpoint in `src/whisperx_api_server/routers/transcriptions.py` already supports both transcription and diarization via request parameters.  No changes are required here. The `diarize` parameter controls diarization, and the existing logic handles it correctly.

### 2.4. Testing Docker Image

A new Dockerfile (`Dockerfile.test`) and a test script (`test_transcription.py`) will be created.

*   **`Dockerfile.test`:**
    *   **Base Image:**  A suitable Python base image (e.g., `python:3.10-slim`).
    *   **Dependencies:** Install necessary testing libraries (e.g., `pytest`, `requests`, and any libraries needed for comparing transcription outputs).  A `requirements-test.txt` file will manage these.
    *   **Copy Files:** Copy the test script (`test_transcription.py`) and any test audio files into the container.
    *   **Entrypoint:** Set the entrypoint to run the tests (e.g., `pytest`).

*   **`test_transcription.py`:**
    *   **Test Cases:** This script will contain several test cases:
        *   **Basic Transcription:**  Test transcription without diarization.
        *   **Transcription with Diarization:** Test transcription with diarization enabled.
        *   **Different Languages (Optional):** If feasible, include tests for different languages.
        *   **Edge Cases:** Test with empty audio files, very short audio files, etc.
    *   **Assertions:** Each test case will:
        1.  Send a request to the `/v1/audio/transcriptions` endpoint of the running `transcribe` service (using the `requests` library). The service will need to be running, likely via `docker compose up`.
        2.  Compare the response with a pre-defined "golden" output.  This comparison might involve:
            *   **Text Comparison:**  Compare the transcribed text (allowing for minor variations if necessary).
            *   **Timestamp Comparison:**  Compare the start and end times of segments and words (allowing for a small tolerance).
            *   **Diarization Comparison:** If diarization is enabled, compare the speaker labels and their corresponding segments.
    *   **Test Audio Files:**  A set of small audio files will be used for testing. These files should be diverse enough to cover different scenarios (e.g., single speaker, multiple speakers, different accents/languages if possible). These will be stored in a `test_data` directory.

## 3. Test Design

### 3.1. Unit Tests

The primary focus of testing will be on the integration between the API and the WhisperX library. Unit tests for individual functions within WhisperX are assumed to be covered by the WhisperX project itself.

### 3.2. Integration Tests (via `test_transcription.py`)

The `transcribe-test` service will be configured to depend on the `transcribe` service, but this only ensures the container is started, not that the service is fully ready to accept requests. To address this:

The `test_transcription.py` script, executed within the `Dockerfile.test` container, will perform the following integration tests:

1.  **Setup:**
    *   Ensure the `transcribe` service is running (likely started beforehand with `docker compose up`).
    *   Potentially pre-download the WhisperX model to the `data/models` directory to speed up the first test run.
    *   The test container will connect to the `transcribe` service using the Docker network.
    *   The test script will implement a retry mechanism to wait for the transcribe service to be fully operational before running tests.

2.  **Test Execution:**
    *   Iterate through the test cases (described in section 2.4).
    *   For each test case:
        *   Construct the appropriate API request (using the `requests` library).
        *   Send the request to the `transcribe` service.
        *   Receive the response.
        *   Compare the response with the expected "golden" output.
    *   The tests will use the internal Docker network to communicate with the `transcribe` service on port 8000.
    *   The test container will be configured with the container name `transcribe-test` for consistency.

3.  **Reporting:**
    *   Use `pytest` to report the results of the tests (pass/fail, error messages).

### 3.3. Test Data

The `whisperx-api-server/test/data` directory will contain:

*   **Audio Files:**  A variety of short audio files (e.g., `single_speaker.m4a`, `multi_speaker.m4a`).
*   **Golden Outputs:** Corresponding "golden" output files (e.g., `single_speaker.json`, `multi_speaker.json`) containing the expected transcription and diarization results. These will be manually created and verified.

## 4. Directory Structure

```text
./
├── data/  <-- Project-level data directory
│   ├── models/       <-- Persistent model storage for whisperx
│   ├── hoarder/      <-- (Existing)
│   ├── immich/       <-- (Existing)
│   ├── ollama/       <-- (Existing)
│   ├── open-webui/   <-- (Existing)
│   ├── owncloud/     <-- (Existing)
│   ├── searxng/      <-- (Existing)
│   ├── ts/           <-- (Existing)
│   ├── vaultwarden/  <-- (Existing)
│   └── windmill/     <-- (Existing)
├── whisperx-api-server/  <-- Cloned repository
│   ├── src/
│   │   └── whisperx_api_server/
│   │       ├── config.py      <-- Modified: download_root
│   │       ├── main.py        <-- No changes
│   │       ├── models.py      <-- No changes
│   │       └── routers/
│   │           └── transcriptions.py  <-- No changes
│   ├── .dockerignore
│   ├── .github/workflows/
│   ├── .gitignore
│   ├── cuda-docker-entrypoint.sh
│   ├── requirements-cpu.txt
│   ├── requirements-cuda.txt
│   ├── requirements.txt
│   ├── Dockerfile.cpu
│   ├── Dockerfile.cuda
│   ├── Dockerfile.test     <-- New: Testing Dockerfile
│   └── test/               <-- New: Test
│       ├── data/
│       │   ├── single_speaker.m4a
│       │   ├── single_speaker.json
│       │   ├── multi_speaker.m4a
│       │   └── multi_speaker.json
│       └── test_transcription.py <-- New: Test script
|       └── requirements-test.txt <-- New: Test dependencies
├── LICENSE
├── README.md         <-- This file (moved to project root)
├── docker-compose.yaml   <-- Modified: Added 'transcribe' service
├── hoarder/              <-- (Existing)
├── immich/               <-- (Existing)
├── ollama/               <-- (Existing)
├── ts-serve/             <-- (Existing)

```
