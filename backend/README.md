# Backend API

FastAPI backend for Fingerprint Blood Group Detection with endpoints for training and prediction.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- GET /health
- GET /version
- GET /api/classes
- GET /api/models
- GET /api/config
- POST /api/train?model=resnet50|vgg16|mobilenet_v2|lenet
- POST /api/predict (multipart/form-data with file)

## Docker

Build and run (dataset mounted read-only):

```bash
cd backend
docker build -t fingerprint-bloodgroup-api .
docker run --rm -p 8000:8000 \
  -e DATA_ROOT=/data/dataset_blood_group \
  -v $(pwd)/../dataset:/data:ro \
  -v $(pwd)/app/storage:/app/app/storage \
  fingerprint-bloodgroup-api
```

or with docker-compose:

```bash
cd backend
docker compose up --build
```