from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Signature Verification API is running!"}
