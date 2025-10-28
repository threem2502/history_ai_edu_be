from pydantic import BaseModel
from typing import Optional

class VisionResponse(BaseModel):
    description: str
    confidence: Optional[float] = None
