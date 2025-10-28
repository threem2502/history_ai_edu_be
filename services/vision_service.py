import os
from uuid import uuid4
from core.config import settings
from schemas.vision import VisionResponse

async def analyze_image(file_bytes: bytes, filename: str) -> VisionResponse:
    # Lưu file tạm (optional, để debug)
    save_dir = settings.UPLOAD_DIR_IMAGE
    os.makedirs(save_dir, exist_ok=True)
    unique_name = f"{uuid4()}_{filename}"
    save_path = os.path.join(save_dir, unique_name)

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    # TODO: gọi model vision thật. Ví dụ:
    # vision_result = real_vision_model.predict(file_bytes)
    # caption = vision_result.caption
    # conf = vision_result.confidence
    # MOCK:
    caption = f"Ảnh '{filename}' đã được nhận và lưu. (Demo mô tả nội dung ảnh ở đây)"
    conf = 0.9

    return VisionResponse(
        description=caption,
        confidence=conf
    )
