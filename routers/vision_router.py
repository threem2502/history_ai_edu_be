from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas.vision import VisionResponse
from services.vision_service import analyze_image

router = APIRouter()

@router.post(
    "/analyze",
    response_model=VisionResponse,
    summary="Nhận diện hình ảnh",
    description="Upload 1 ảnh (multipart/form-data) để được mô tả nội dung."
)
async def vision_analyze(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là hình ảnh.")
    img_bytes = await image.read()
    result = await analyze_image(img_bytes, image.filename)
    return result
