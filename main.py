from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

class Comment(BaseModel):
    comment: str

class Email(BaseModel):
    email: str
app = FastAPI()

# Load the trained model
model = joblib.load('model.pkl')
email_model=joblib.load('')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake Comment Detector API"}

@app.post("/detect-fake-comment")
async def detect_fake_comment(comment: Comment):
    try:
        prediction = model.predict([comment.comment])[0]
        result = "Fake Comment" if prediction == 1 else "Real Comment"
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/detect-spam-email")
async def detect_spam_email(email: Email):
    try:
        prediction = email_model.predict([email.email])[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

