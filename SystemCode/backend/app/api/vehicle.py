from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.dependencies import get_db
from app.schemas.vehicle import (
    VehicleActionListResponse,
    VehicleDetailResponse,
    VehicleListResponse,
    VehicleMaintenanceResponse,
    VehicleMaintenanceUpdate,
)
from app.services.vehicle_service import (
    VehicleQueryError,
    VehicleService,
    parse_vehicle_id,
)


router = APIRouter(
    prefix="/vehicles",
    tags=["Vehicles"],
)


@router.get(
    "",
    response_model=VehicleListResponse,
    response_model_by_alias=True,
    summary="List vehicles by latest risk analysis",
)
def list_vehicles(
    q: str | None = Query(
        default=None,
        description="Vehicle id search, such as 89321 or VHC-89321.",
    ),
    predicted_class: str | None = Query(
        default=None,
        description="Latest analysis action. Omit it to include every action.",
    ),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> VehicleListResponse:
    try:
        return VehicleService.list_vehicles(
            db,
            q=q,
            predicted_class=predicted_class,
            page=page,
            page_size=page_size,
        )
    except VehicleQueryError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc


@router.get(
    "/actions",
    response_model=VehicleActionListResponse,
    response_model_by_alias=True,
    summary="List analysis actions, most urgent first",
)
def list_vehicle_actions(
    db: Session = Depends(get_db),
) -> VehicleActionListResponse:
    return VehicleService.list_actions(db)


@router.patch(
    "/{vehicle_id}/maintenance",
    response_model=VehicleMaintenanceResponse,
    response_model_by_alias=True,
    summary="Set whether a vehicle needs maintenance",
)
def set_vehicle_maintenance(
    vehicle_id: str,
    body: VehicleMaintenanceUpdate,
    db: Session = Depends(get_db),
) -> VehicleMaintenanceResponse:
    parsed = parse_vehicle_id(vehicle_id)
    if parsed is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Vehicle id '{vehicle_id}' is not valid. "
                "Use a number or VHC-<number>."
            ),
        )

    updated = VehicleService.set_maintenance(
        db,
        parsed,
        body.in_maintenance,
    )
    if updated is None:
        raise HTTPException(
            status_code=404,
            detail=f"Vehicle '{vehicle_id}' was not found.",
        )

    return VehicleMaintenanceResponse(
        vehicle_id=vehicle_id if vehicle_id.upper().startswith("VHC-") else f"VHC-{parsed}",
        in_maintenance=updated,
    )


@router.get(
    "/{vehicle_id}",
    response_model=VehicleDetailResponse,
    response_model_by_alias=True,
    summary="Get one vehicle risk detail",
)
def get_vehicle(
    vehicle_id: str,
    db: Session = Depends(get_db),
) -> VehicleDetailResponse:
    parsed = parse_vehicle_id(vehicle_id)
    if parsed is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Vehicle id '{vehicle_id}' is not valid. "
                "Use a number or VHC-<number>."
            ),
        )

    return VehicleService.get_vehicle(db, parsed)
