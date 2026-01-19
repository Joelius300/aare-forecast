# NOTE: This is a semi-temporary, kinda weird setup because the flow forecasts come from BAFU and were patched
# into this whole thing as an afterthought without wanting to waste time to make it clean. Could use refactoring.
from fastapi import APIRouter

router = APIRouter()

@router.get("/forecast/flow", description=API_DESC, response_model=ForecastPayload)
async def get_flow_forecasts():
    pass


