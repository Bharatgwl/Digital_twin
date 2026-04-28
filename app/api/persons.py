import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.person import Person
from app.services.level_3_recognition.recognizer import FaceRecognizer

router = APIRouter(prefix="/persons", tags=["Person Identity"])

recognizer = FaceRecognizer()


@router.post("/register")
async def register_person(
    name: str = Form(...),
    age: str = Form(None),
    gender: str = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Register a new person by uploading a face image.
    The face embedding is extracted and stored in the database.
    """
    # Read uploaded image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        person = recognizer.register_person(
            name=name, frame=frame, db=db, age=age, gender=gender
        )
        return {
            "message": f"Person '{name}' registered successfully.",
            "person_id": person.id,
            "name": person.name,
            "age": person.age,
            "gender": person.gender,
            "created_at": str(person.created_at),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/")
def list_persons(db: Session = Depends(get_db)):
    """List all registered persons in the database."""
    persons = db.query(Person).all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "gender": p.gender,
            "sighting_count": p.sighting_count,
            "created_at": str(p.created_at),
            "last_seen_at": str(p.last_seen_at),
        }
        for p in persons
    ]


@router.get("/{person_id}")
def get_person(person_id: int, db: Session = Depends(get_db)):
    """Get details of a specific person by ID."""
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")
    return {
        "id": person.id,
        "name": person.name,
        "age": person.age,
        "gender": person.gender,
        "face_image_path": person.face_image_path,
        "sighting_count": person.sighting_count,
        "created_at": str(person.created_at),
        "last_seen_at": str(person.last_seen_at),
    }


@router.delete("/{person_id}")
def delete_person(person_id: int, db: Session = Depends(get_db)):
    """Delete a person from the database."""
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")
    db.delete(person)
    db.commit()
    return {"message": f"Person '{person.name}' deleted successfully."}
