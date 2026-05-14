from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas.snapshot import PRODUCT_LABEL_EMPTY_VALUE, ProductLabelImage, product_label_image_stems

# app/api/endpoints/images.py → parents[2] = app/
_STATIC = Path(__file__).resolve().parents[2] / "static"
_CACHE = {"Cache-Control": "public, max-age=3600"}

router = APIRouter(prefix="/images")
router.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@router.get("/food_item/{food_item_id}")
def get_food_item_image(food_item_id: str) -> FileResponse:
    """Serve a food item WebP from static/food_items/."""
    path = _STATIC / "food_items" / f"{Path(food_item_id).name}.webp"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Food item image not found")
    return FileResponse(path, media_type="image/webp", headers=_CACHE)


@router.get("/product_label/{label_name}")
def get_product_label_image(label_name: ProductLabelImage) -> FileResponse:
    """Serve a product label WebP from static/product_labels/."""
    stem = str(label_name)
    if not product_label_image_stems() and stem == PRODUCT_LABEL_EMPTY_VALUE:
        raise HTTPException(status_code=404, detail="No product label images configured.")
    path = _STATIC / "product_labels" / f"{stem}.webp"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Product label image not found")
    return FileResponse(path, media_type="image/webp", headers=_CACHE)
