import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer



# Example training data
X_train = ["Sample comment 1", "Sample comment 2"]
y_train = [1, 0]

# Fit the vectorizer
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)

# Train the model
model = RandomForestClassifier()
model.fit(X_train_transformed, y_train)

# Save the vectorizer and model
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')


class Comment(BaseModel):
    comment: str

class Email(BaseModel):
    email: str

app = FastAPI()

# Load the trained model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')
email_model = joblib.load('email-model.pkl')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake Comment Detector API"}

@app.post("/detect-fake-comment")
async def detect_fake_comment(comment: Comment):
    try:
        # Preprocess the input comment
        comment_vector = vectorizer.transform([comment.comment])
        prediction = model.predict(comment_vector)[0]
        result = "Fake Comment" if prediction == 1 else "Real Comment"
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-spam-email")
async def detect_spam_email(email: Email):
    try:
        # Preprocess the input email
        email_vector = vectorizer.transform([email.email])
        prediction = email_model.predict(email_vector)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
