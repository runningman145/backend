# Backend API - Vehicle Detection & Tracking System

This is the backend API for a vehicle detection and tracking system using camera networks. The project uses Flask to provide REST endpoints for managing cameras, detections, and user authentication with JWT tokens.

## Project Overview

The system enables:
- User registration and authentication with JWT
- Camera management (create, read, update, delete)
- Vehicle detection logging from ML models
- Media uploads (pictures and videos)
- Real-time camera status tracking
- Comprehensive health check endpoints

## Tech Stack

- **Framework**: Flask 2.3+
- **Authentication**: JWT (PyJWT)
- **Database**: SQLite
- **Environment**: Python 3.8+
- **File Upload**: Werkzeug

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/runningman145/backend
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -e .
```

4. **Install development dependencies (optional)**
```bash
pip install -e ".[dev]"
```

5. **Create `.env` file with secret key**
```bash
echo 'SECRET_KEY=your_super_secret_key_change_in_production' > .env
```

6. **Initialize the database**
```bash
flask --app api init-db
```

## Running the Application

### Development Mode
```bash
export FLASK_APP=api
export FLASK_ENV=development
flask run
```

The API will be available at `http://localhost:5000`

## API Endpoints


### Cameras

- **GET** `/cameras` - List all cameras
- **POST** `/cameras` - Create a new camera
- **GET** `/cameras/<camera_id>` - Get camera details
- **PUT** `/cameras/<camera_id>` - Update camera (name, location, status)
- **DELETE** `/cameras/<camera_id>` - Delete a camera

Example camera creation:
```bash
curl -X POST http://localhost:5000/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Front Gate",
    "latitude": "40.7128",
    "longitude": "-74.0060",
    "status": "online"
  }'
```

### Detections & Tracking

- **GET** `/tracking/detections` - Get all detections with camera info
- **POST** `/tracking/detections` - Log a new detection
- **POST** `/tracking/upload` - Upload media (picture/video)

Supported media formats: JPG, JPEG, PNG, GIF, MP4, AVI, MOV, MKV, WebM

Example upload:
```bash
curl -X POST http://localhost:5000/tracking/upload \
  -F "file=@image.jpg" \
  -F "camera_id=camera-123" \
  -F "detection_id=detection-456"
```

### Health Checks

- **GET** `/health` - Basic health check
- **GET** `/health/ready` - Readiness check (verifies DB connectivity)
- **GET** `/health/live` - Liveness check
- **GET** `/health/ping` - Simple ping
- **GET** `/health/status` - Detailed status with database stats


## Project Structure

```
backend/
├── api/
│   ├── __init__.py           # Flask app initialization
│   ├── auth.py               # Authentication endpoints
│   ├── cameras.py            # Camera management endpoints
│   ├── tracking.py           # Detection and upload endpoints
│   ├── health.py             # Health check endpoints
│   ├── db.py                 # Database connection
│   ├── model.py              # Pytorch model
│   ├── schema.sql            # Database schema
│   └── __pycache__/
├── instance/
│   └── uploads/              # Uploaded media files
├── .env                      # Environment variables (not in git)
├── .gitignore
├── pyproject.toml            # Project metadata and dependencies
└── README.md
```

## Database

The project uses SQLite with three main tables:

- **users** - User accounts
- **cameras** - Camera devices with location and status
- **detections** - Vehicle detections recorded by ML models

Initialize database:
```bash
flask --app api init-db
```

## Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your_production_secret_key_here
FLASK_ENV=production
FLASK_DEBUG=False
```

## Error Handling

The API returns standard HTTP status codes:
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized (authentication required)
- `403` - Forbidden (permission denied)
- `404` - Not Found
- `409` - Conflict (resource already exists)
- `500` - Server Error
- `503` - Service Unavailable

## References

- [Flask documentation](https://flask.palletsprojects.com/)
- [SQLite documentation](https://sqlite.org/docs.html)
- [PyJWT documentation](https://pyjwt.readthedocs.io/)
- [Werkzeug documentation](https://werkzeug.palletsprojects.com/)

