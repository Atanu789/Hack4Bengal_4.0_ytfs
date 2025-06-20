# My FastAPI Project

A modern FastAPI project with clean architecture and best practices.

## Features

- ✅ FastAPI with async/await support
- ✅ Pydantic v2 for data validation
- ✅ Clean project structure
- ✅ Environment-based configuration
- ✅ CORS middleware
- ✅ API versioning
- ✅ Comprehensive error handling
- ✅ Security utilities
- ✅ Mock CRUD operations

## Installation

1. Create and activate virtual environment:
```bash
python -m venv fastapi-env
source fastapi-env/bin/activate  # On Windows: fastapi-env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the Application

### Development
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/api/v1/openapi.json

## Project Structure

```
my-fastapi-project/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application factory
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── endpoints/      # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration settings
│   │   └── security.py        # Security utilities
│   ├── models/                # Database models (SQLAlchemy)
│   ├── schemas/               # Pydantic schemas
│   └── services/              # Business logic
├── tests/                     # Test files
├── requirements.txt
├── .env                       # Environment variables
└── README.md
```

## Available Endpoints

### Items
- `GET /api/v1/items/` - List all items
- `GET /api/v1/items/{item_id}` - Get item by ID
- `POST /api/v1/items/` - Create new item
- `PUT /api/v1/items/{item_id}` - Update item
- `DELETE /api/v1/items/{item_id}` - Delete item

### Users
- `GET /api/v1/users/` - List all users
- `GET /api/v1/users/{user_id}` - Get user by ID
- `POST /api/v1/users/` - Create new user
- `PUT /api/v1/users/{user_id}` - Update user

### Health Check
- `GET /health` - Application health status

## Testing

Run tests with pytest:
```bash
pytest
```

# YouTube Service Integration Guide

## Project Structure

Create the following files in your FastAPI project:

```
your_project/
├── app/
│   ├── __init__.py
│   ├── main.py (your existing file - update it)
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── api.py (new)
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           └── youtube.py (new)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── youtube.py (new)
│   ├── services/
│   │   ├── __init__.py
│   │   └── youtube_service.py (the original service file)
│   └── core/
│       ├── __init__.py
│       └── config.py (your existing config)
```

## Installation

Add these dependencies to your requirements.txt:

```txt
aiohttp==3.8.6
```

Then install:
```bash
pip install aiohttp
```

## Testing in Postman

### Base URL
Assuming your `settings.API_V1_STR` is `/api/v1`, your endpoints will be:

- Base URL: `http://localhost:8000/api/v1/youtube`

### Endpoint Tests

#### 1. Get Video Info (POST)
- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/youtube/info`
- **Headers**: `Content-Type: application/json`
- **Body** (raw JSON):
```json
{
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "quality": "720p"
}
```

#### 2. Get Video Info by ID (GET)
- **Method**: `GET`
- **URL**: `http://localhost:8000/api/v1/youtube/info/dQw4w9WgXcQ`
- **Query Parameters**: 
  - `quality`: `highest` (optional)

#### 3. Get All Formats (POST)
- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/youtube/formats`
- **Headers**: `Content-Type: application/json`
- **Body** (raw JSON):
```json
{
    "url": "https://youtu.be/dQw4w9WgXcQ"
}
```

## Sample Postman Collection

### Environment Variables
Create a Postman environment with:
- `base_url`: `http://localhost:8000`
- `api_version`: `/api/v1`
- `youtube_url`: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`

### Pre-request Script (Optional)
Add this to collection pre-request script for logging:
```javascript
console.log("Testing YouTube API endpoints");
console.log("Base URL:", pm.environment.get("base_url"));
```

### Test Script Template
Add this to your request tests:
```javascript
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response is JSON", function () {
    pm.response.to.be.json;
});

pm.test("Response has success field", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('success');
});

pm.test("Success is true", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.success).to.be.true;
});

pm.test("Has video information when successful", function () {
    var jsonData = pm.response.json();
    if(jsonData.success) {
        pm.expect(jsonData).to.have.property('video_id');
        pm.expect(jsonData).to.have.property('title');
        pm.expect(jsonData).to.have.property('author');
    }
});
```

## API Documentation

Once your server is running, you can access:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

The YouTube endpoints will be organized under the "YouTube" tag in the documentation.

## Error Handling

The API returns proper HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid URL, video not found)
- `404`: Video not found
- `500`: Internal server error

## Security Notes

1. **Rate Limiting**: Consider adding rate limiting for production
2. **Input Validation**: URLs are validated for YouTube domains
3. **Error Logging**: Add proper logging for debugging
4. **CORS**: Already configured in your main.py

## Sample Response

```json
{
  "success": true,
  "video_id": "dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up",
  "description": "The official video for...",
  "duration": 213,
  "view_count": 1234567890,
  "author": "Rick Astley",
  "upload_date": "2009-10-25",
  "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
  "best_format": {
    "quality": "hd720",
    "resolution": "720p",
    "ext": "mp4",
    "url": "https://...",
    "filesize": 12345678
  },
  "all_formats": [...]
}
```