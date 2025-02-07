import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
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


# Example training data for spam email detection
X_email_train = [
    "This is a spam email",
    "Buy now, limited time offer",
    "Congratulations, you won a prize",
    "This is a regular email",
    "Let's schedule a meeting",
    "Here's the report you requested"
]
y_email_train = [1, 0, 1, 1, 0, 1]  # 1 for Spam, 0 for Not Spam

# Fit the vectorizer
email_vectorizer = TfidfVectorizer()
X_email_train_transformed = email_vectorizer.fit_transform(X_email_train)

# Train the spam email model
email_model = RandomForestClassifier()
email_model.fit(X_email_train_transformed, y_email_train)

# Save the email vectorizer and model
joblib.dump(email_vectorizer, 'email_vectorizer.pkl')
joblib.dump(email_model, 'email_model.pkl')

class Comment(BaseModel):
    comment: str

class Email(BaseModel):
    email: str

app = FastAPI()

# Load the trained models and vectorizers
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')
email_vectorizer = joblib.load('email_vectorizer.pkl')
email_model = joblib.load('email_model.pkl')

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
        email_vector = email_vectorizer.transform([email.email])
        prediction = email_model.predict(email_vector)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

