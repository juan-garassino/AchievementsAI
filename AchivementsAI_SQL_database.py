"""
database.py

This module sets up the SQLAlchemy ORM for a PostgreSQL database containing personal information.

Key components:
1. Database connection setup using environment variables
2. SQLAlchemy engine and session creation
3. PersonalInfo model definition
4. Function to get a database session
5. Sample data and function to populate the database

Usage:
- Set environment variables for DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
- Run this script directly to create tables and populate with sample data
- Import get_db and PersonalInfo in other modules to interact with the database

Note: Modify the sample_data list to include your actual personal information.
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database connection parameters
DB_NAME = "juan_personal_db"
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")

# Create the database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a base class for declarative models
Base = declarative_base()

# Define the PersonalInfo model
class PersonalInfo(Base):
    __tablename__ = 'personal_info'

    id = Column(Integer, primary_key=True)
    category = Column(String(50))
    info_key = Column(String(100))
    info_value = Column(String)
    date_associated = Column(Date)

# Create the database and tables
Base.metadata.create_all(engine)

# Create a session to interact with the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Sample data - modify this to fit your actual information
sample_data = [
    ("Education", "University", "Technical University of Munich", "2018-09-01"),
    ("Education", "Degree", "Master of Science in Computer Science", "2020-06-30"),
    ("Work Experience", "Le Wagon", "Data Science Instructor", "2021-01-15"),
    ("Work Experience", "Freelance", "Machine Learning Engineer", "2022-03-01"),
    ("Skills", "Programming Languages", "Python, JavaScript, SQL", None),
    ("Skills", "Frameworks", "TensorFlow, PyTorch, React", None),
    ("Personal", "Languages", "English (Fluent), Spanish (Native), German (Intermediate)", None),
    ("Personal", "Date of Birth", "1990-05-15", "1990-05-15"),
    ("Achievements", "Published Paper", "Deep Learning in Architectural Design", "2022-11-10"),
    ("Projects", "deepTechno", "Music synthesis using transformer architecture", "2023-02-01"),
]

# Function to populate the database with sample data
def populate_db():
    db = SessionLocal()
    for category, key, value, date in sample_data:
        info = PersonalInfo(
            category=category,
            info_key=key,
            info_value=value,
            date_associated=datetime.strptime(date, "%Y-%m-%d").date() if date else None
        )
        db.add(info)
    db.commit()
    db.close()
    print("Database populated with sample data")

if __name__ == "__main__":
    populate_db()