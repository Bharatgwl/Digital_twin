from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func

from app.core.database import Base


class Person(Base):
    """Stores identity profiles for recognized persons."""
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, index=True)
    
    # Face embedding stored as JSON string (list of floats)
    face_embedding = Column(Text, nullable=False)
    
    # Path to the stored face image
    face_image_path = Column(String(500), nullable=True)
    
    # Additional features / metadata
    age = Column(String(20), nullable=True)
    gender = Column(String(20), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_seen_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # How many times this person has been spotted
    sighting_count = Column(Integer, default=1)

    def __repr__(self):
        return f"<Person(id={self.id}, name='{self.name}', sightings={self.sighting_count})>"
