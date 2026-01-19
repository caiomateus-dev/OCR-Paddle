import io
import logging
import os
import time
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image
from starlette.concurrency import run_in_threadpool

# Avoid PaddleOCR trying to check model hosters at import time (network is optional).
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR

logger = logging.getLogger("teste-validator")


class OcrItem(BaseModel):
    text: str
    bbox: list[list[int]]


class OcrResponse(BaseModel):
    items: list[OcrItem]


def _load_ocr_config() -> dict[str, Any]:
    pyproject_path = Path(__file__).with_name("pyproject.toml")
    if not pyproject_path.exists():
        return {}

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    tool_cfg = data.get("tool", {})
    app_cfg = tool_cfg.get("teste_validator", {})
    return dict(app_cfg.get("ocr", {}) or {})


@lru_cache(maxsize=1)
def _get_ocr() -> PaddleOCR:
    import inspect

    cfg = _load_ocr_config()

    if cfg.get("disable_model_source_check", False):
        os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    # Pedido: angle classification SIM, unwarping NÃO.
    cfg["use_doc_unwarping"] = False
    cfg["use_doc_orientation_classify"] = False
    cfg["use_textline_orientation"] = True

    # Backwards-compat with older config naming.
    if "rec_image_shape" in cfg and "text_rec_input_shape" not in cfg:
        cfg["text_rec_input_shape"] = cfg["rec_image_shape"]

    def _parse_shape(v: Any) -> Any:
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",") if p.strip()]
            if parts and all(p.lstrip("-").isdigit() for p in parts):
                return [int(p) for p in parts]
        return v

    if "text_rec_input_shape" in cfg:
        cfg["text_rec_input_shape"] = _parse_shape(cfg["text_rec_input_shape"])
    if "text_det_input_shape" in cfg:
        cfg["text_det_input_shape"] = _parse_shape(cfg["text_det_input_shape"])

    # Defaults "top" for server Mac potente (CPU): balanceia qualidade/velocidade.
    # Você pode sobrescrever tudo via `[tool.teste_validator.ocr]` no `pyproject.toml`.
    cfg.setdefault("ocr_version", "PP-OCRv5")
    cfg.setdefault("lang", "pt")
    cfg.setdefault("text_det_limit_type", "max")
    cfg.setdefault("text_det_limit_side_len", 1536)
    cfg.setdefault("text_det_thresh", 0.3)
    cfg.setdefault("text_det_box_thresh", 0.6)
    cfg.setdefault("text_det_unclip_ratio", 2.0)
    cfg.setdefault("text_rec_score_thresh", 0.0)
    cfg.setdefault("textline_orientation_batch_size", 8)
    cfg.setdefault("text_recognition_batch_size", 8)

    # PaddleOCR pipeline is strict: unknown kwargs raise ValueError.
    sig = inspect.signature(PaddleOCR.__init__)
    allowed_keys = {k for k in sig.parameters.keys() if k not in {"self", "kwargs"}}
    init_kwargs = {k: v for k, v in cfg.items() if k in allowed_keys}

    return PaddleOCR(**init_kwargs)


def _bytes_to_bgr_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    rgb = np.asarray(img)
    # PaddleOCR accepts RGB/BGR; keep BGR to match common cv2 pipelines.
    return rgb[:, :, ::-1].copy()


def _poly_to_bbox(poly: Any) -> list[list[int]]:
    arr = np.asarray(poly)
    if arr.shape != (4, 2):
        arr = arr.reshape(4, 2)
    return [[int(x), int(y)] for x, y in arr.tolist()]


def _extract_items(pred: Any) -> list[OcrItem]:
    items: list[OcrItem] = []

    # PaddleOCR >=2.7 returns list[dict] with rec_polys + rec_texts.
    if isinstance(pred, list) and pred and isinstance(pred[0], dict):
        for page in pred:
            texts = page.get("rec_texts") or []
            polys = page.get("rec_polys") or []
            for text, poly in zip(texts, polys):
                if not text:
                    continue
                items.append(OcrItem(text=str(text), bbox=_poly_to_bbox(poly)))
        return items

    # Legacy format: list[list[[poly, (text, score)]]]
    if isinstance(pred, list):
        for line in pred:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            poly, rec = line[0], line[1]
            text = rec[0] if isinstance(rec, (list, tuple)) and rec else ""
            if not text:
                continue
            items.append(OcrItem(text=str(text), bbox=_poly_to_bbox(poly)))

    return items


def _mem_snapshot() -> dict[str, int]:
    snap: dict[str, int] = {}
    try:
        import psutil  # type: ignore

        snap["rss_bytes"] = int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        import resource

        snap["ru_maxrss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        pass
    return snap


def _mem_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in ("rss_bytes", "ru_maxrss_kb"):
        if k in before and k in after:
            out[k] = int(after[k]) - int(before[k])
    return out


app = FastAPI(title="teste-validator")


@app.on_event("startup")
async def _startup() -> None:
    # Preload OCR model once when API goes online (no load per request).
    t0 = time.perf_counter()
    mem0 = _mem_snapshot()
    _get_ocr()
    mem1 = _mem_snapshot()
    logger.info(
        "OCR model loaded in %.2fs mem=%s delta=%s",
        time.perf_counter() - t0,
        mem1,
        _mem_delta(mem0, mem1),
    )


@app.post("/parse/image", response_model=OcrResponse)
async def parse_image(file: UploadFile = File(...)) -> OcrResponse:
    t_req = time.perf_counter()
    mem0 = _mem_snapshot()
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    ocr = _get_ocr()
    t_decode = time.perf_counter()
    img = _bytes_to_bgr_image(image_bytes)
    decode_s = time.perf_counter() - t_decode

    # Overrides por request para evitar cortes/resize agressivo quando necessário.
    pred_kwargs: dict[str, Any] = {
        "use_doc_unwarping": False,
        "use_doc_orientation_classify": False,
        "use_textline_orientation": True,
    }
    pred = await run_in_threadpool(lambda: ocr.predict(img, **pred_kwargs))
    items = _extract_items(pred)

    mem1 = _mem_snapshot()
    logger.info(
        "OCR request filename=%s content_type=%s bytes=%d decode=%.3fs total=%.3fs mem=%s delta=%s items=%d",
        file.filename,
        file.content_type,
        len(image_bytes),
        decode_s,
        time.perf_counter() - t_req,
        mem1,
        _mem_delta(mem0, mem1),
        len(items),
    )
    return OcrResponse(items=items)
