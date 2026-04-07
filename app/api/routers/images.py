from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.schemas.product_label_image import (
    PRODUCT_LABEL_EMPTY_VALUE,
    ProductLabelImage,
    product_label_image_stems,
)

_STATIC = Path(__file__).resolve().parents[2] / "static"
_CACHE = {"Cache-Control": "public, max-age=3600"}

router = APIRouter(prefix="/images")
router.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@router.get("/food_item/{image_id}")
def get_food_item_image(image_id: str) -> FileResponse:
    """Serve a food item image JPEG from static image directory."""
    path = _STATIC / "food_item_images" / f"{Path(image_id).name}.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg", headers=_CACHE)


@router.get("/product_label/{label_name}")
def get_product_label_image(label_name: ProductLabelImage) -> FileResponse:
    """Serve a product label webp from static image directory; ``label_name`` must be a :class:`ProductLabelImage` value."""
    stem = str(label_name)
    if not product_label_image_stems() and stem == PRODUCT_LABEL_EMPTY_VALUE:
        raise HTTPException(status_code=404, detail="No product label images configured.")
    path = _STATIC / "product_label_images" / f"{stem}.webp"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/webp", headers=_CACHE)
