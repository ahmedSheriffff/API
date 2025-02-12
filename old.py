import joblib
import logging
import os
from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Security: API Key
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY must be set in environment variables.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate Limiting to prevent abuse (10 requests per minute per user)
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI
app = FastAPI()

# Register rate limiter exception handler
app.state.limiter = limiter
app.add_exception_handler(_rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Enable CORS (Allow frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models and vectorizers
model_files = ["vectorizer.pkl", "model.pkl", "email_vectorizer.pkl", "email_model.pkl"]
models = {}

for file in model_files:
    try:
        models[file] = joblib.load(file)
        logger.info(f"{file} loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Missing model file: {file}")
        raise RuntimeError(f"{file} is missing. Ensure all .pkl files are present.")
    except Exception as e:
        logger.error(f"Error loading {file}: {e}")
        raise RuntimeError("Failed to load machine learning models.")

# Input models
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=3, max_length=500, description="Text of the comment")

class EmailRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=1000, description="Email content")

# API Key Validation
def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Fake Comment & Spam Email Detector API"}

@app.post("/detect-fake-comment", tags=["Fake Comment Detection"])
@limiter.limit("10/minute")
async def detect_fake_comment(request: Request, comment: CommentRequest, api_key: str = Depends(verify_api_key)):
    try:
        if not API_KEY:
            raise ValueError("API_KEY must be set in environment variables.")
        comment_vector = models["vectorizer.pkl"].transform([comment.comment])
        prediction = models["model.pkl"].predict(comment_vector)[0]
        result = "Fake Comment" if prediction == 1 else "Real Comment"
        return {"result": result}
    except Exception as e:
        logger.error(f"Error detecting fake comment: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/detect-spam-email", tags=["Spam Email Detection"])
@limiter.limit("10/minute")
async def detect_spam_email(request: Request, email: EmailRequest, api_key: str = Depends(verify_api_key)):
    try:
        if not API_KEY:
            raise ValueError("API_KEY must be set in environment variables.")
        email_vector = models["email_vectorizer.pkl"].transform([email.email])
        prediction = models["email_model.pkl"].predict(email_vector)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return {"result": result}
    except Exception as e:
        logger.error(f"Error detecting spam email: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
