from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pymongo.database import Database

from app.core.mongo import get_db
from app.services.digital_twin.repository import DigitalTwinRepository
from app.utils.serialization import to_jsonable


router = APIRouter(tags=["Digital Twins"])


def _repository(db: Database = Depends(get_db)) -> DigitalTwinRepository:
    return DigitalTwinRepository(db)


@router.get("/digital-twins")
def list_digital_twins(
    q: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    repo: DigitalTwinRepository = Depends(_repository),
):
    query = {}
    if q:
        query = {
            "$or": [
                {"person_id": {"$regex": q, "$options": "i"}},
                {"display_name": {"$regex": q, "$options": "i"}},
                {"dominant_clothing_color": {"$regex": q, "$options": "i"}},
                {"last_camera_id": {"$regex": q, "$options": "i"}},
            ]
        }

    profiles = list(
        repo.profiles.find(
            query,
            {"_id": 0, "face_embedding": 0},
        )
        .sort("last_seen_at", -1)
        .limit(limit)
    )
    return to_jsonable(profiles)


@router.get("/digital-twins/{person_id}")
def get_digital_twin(person_id: str, repo: DigitalTwinRepository = Depends(_repository)):
    profile = repo.profiles.find_one({"person_id": person_id}, {"_id": 0, "face_embedding": 0})
    if profile is None:
        raise HTTPException(status_code=404, detail="Digital twin not found.")

    recent_observation = repo.observations.find_one(
        {"person_id": person_id},
        sort=[("captured_at", -1)],
        projection={"_id": 0},
    )
    recent_visit = repo.visits.find_one(
        {"person_id": person_id},
        sort=[("entry_time", -1)],
        projection={"_id": 0},
    )

    return to_jsonable(
        {
            "profile": profile,
            "recent_observation": recent_observation,
            "recent_visit": recent_visit,
        }
    )


@router.get("/digital-twins/{person_id}/timeline")
def get_timeline(
    person_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    repo: DigitalTwinRepository = Depends(_repository),
):
    timeline = list(
        repo.observations.find({"person_id": person_id}, {"_id": 0})
        .sort("captured_at", -1)
        .limit(limit)
    )
    return to_jsonable(timeline)


@router.get("/digital-twins/{person_id}/trajectory")
def get_trajectory(
    person_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    repo: DigitalTwinRepository = Depends(_repository),
):
    trajectory = list(
        repo.observations.find(
            {"person_id": person_id},
            {"_id": 0, "captured_at": 1, "center": 1, "camera_id": 1},
        )
        .sort("captured_at", -1)
        .limit(limit)
    )
    return to_jsonable(trajectory)


@router.get("/digital-twins/{person_id}/visits")
def get_visits(
    person_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    repo: DigitalTwinRepository = Depends(_repository),
):
    visits = list(
        repo.visits.find({"person_id": person_id}, {"_id": 0})
        .sort("entry_time", -1)
        .limit(limit)
    )
    return to_jsonable(visits)


@router.get("/stats/live")
def live_stats(
    request: Request,
    repo: DigitalTwinRepository = Depends(_repository),
):
    runtime_stats = request.app.state.runtime.get_live_stats()
    total_profiles = repo.profiles.count_documents({})
    total_observations = repo.observations.count_documents({})
    return {
        **runtime_stats,
        "total_profiles": total_profiles,
        "total_observations": total_observations,
    }
