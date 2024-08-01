"""
main.py

This module creates a FastAPI application that provides a RESTful API for interacting with
the personal information database.

Key components:
1. FastAPI application setup
2. Pydantic models for request and response validation
3. CRUD operations for personal information:
   - Create a new entry (POST /info/)
   - Read all entries (GET /info/)
   - Read entry by ID (GET /info/{info_id})
   - Read entries by category (GET /info/category/{category})

Usage:
- Ensure database.py is set up and the database is populated
- Run this script to start the FastAPI server
- Access the API documentation at http://localhost:8000/docs

Note: This API allows interaction with the personal information database through
HTTP requests, making it easy to integrate with front-end applications or other services.
"""

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
from database import get_db, PersonalInfo

app = FastAPI()

class PersonalInfoBase(BaseModel):
    category: str
    info_key: str
    info_value: str
    date_associated: Optional[date] = None

class PersonalInfoCreate(PersonalInfoBase):
    pass

class PersonalInfoResponse(PersonalInfoBase):
    id: int

    class Config:
        orm_mode = True

@app.post("/info/", response_model=PersonalInfoResponse)
def create_info(info: PersonalInfoCreate, db: Session = Depends(get_db)):
    db_info = PersonalInfo(**info.dict())
    db.add(db_info)
    db.commit()
    db.refresh(db_info)
    return db_info

@app.get("/info/", response_model=List[PersonalInfoResponse])
def read_info(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    info = db.query(PersonalInfo).offset(skip).limit(limit).all()
    return info

@app.get("/info/{info_id}", response_model=PersonalInfoResponse)
def read_info_by_id(info_id: int, db: Session = Depends(get_db)):
    info = db.query(PersonalInfo).filter(PersonalInfo.id == info_id).first()
    if info is None:
        raise HTTPException(status_code=404, detail="Info not found")
    return info

@app.get("/info/category/{category}", response_model=List[PersonalInfoResponse])
def read_info_by_category(category: str, db: Session = Depends(get_db)):
    info = db.query(PersonalInfo).filter(PersonalInfo.category == category).all()
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)