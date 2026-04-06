from fastapi import FastAPI
import uvicorn
from client import FairAuditEnv

app = FastAPI()
env_instance = FairAuditEnv()

@app.get("/")
def read_root():
    return {"message": "FairAudit Environment is online. API is ready."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()