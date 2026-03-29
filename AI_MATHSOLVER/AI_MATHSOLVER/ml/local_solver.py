"""
Handwritten / printed math image → text (RapidOCR + ONNX) → SymPy solve/simplify.
Tuned for linear and quadratic equations (superscripts, longer lines, looser OCR when needed).
"""

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import sympy as sp
from PIL import Image, ImageOps
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_ocr_engine = None

_MAX_OCR_DIM = 2000
_MIN_SHORT_EDGE_OCR = 160


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR(print_verbose=False)
    return _ocr_engine


def _resize_rgb(img: Image.Image) -> np.ndarray:
    w, h = img.size
    longest = max(w, h)
    if longest > _MAX_OCR_DIM:
        scale = _MAX_OCR_DIM / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return np.array(img)


def _ensure_min_size(rgb: np.ndarray) -> np.ndarray:
    """RapidOCR often misses thin/wide equation strips; upsample short edge."""
    h, w = rgb.shape[:2]
    m = min(h, w)
    if m < _MIN_SHORT_EDGE_OCR:
        scale = _MIN_SHORT_EDGE_OCR / m
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return rgb


def _enhance_for_math_ocr(rgb: np.ndarray) -> np.ndarray:
    rgb = _ensure_min_size(rgb)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    sharp = cv2.addWeighted(g, 1.65, blur, -0.65, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)


def _invert_if_dark(rgb: np.ndarray) -> np.ndarray:
    """If the page is mostly dark (photo of screen), invert for OCR."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if float(np.mean(gray)) < 120:
        rgb = cv2.bitwise_not(rgb)
    return rgb


def _autocontrast(rgb: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(rgb)
    pil = ImageOps.autocontrast(pil, cutoff=2)
    return np.array(pil)


def _box_sort_key(item) -> tuple:
    box = item[0]
    if box is None or len(box) < 1:
        return (0.0, 0.0)
    arr = np.array(box, dtype=np.float32)
    return (float(np.min(arr[:, 1])), float(np.min(arr[:, 0])))


def _join_ocr_tokens(result) -> str:
    if not result:
        return ""
    ordered = sorted(result, key=_box_sort_key)
    parts = []
    for item in ordered:
        if item and len(item) > 1:
            t = str(item[1]).strip()
            if t:
                parts.append(t)
    return "".join(parts)


def _math_ocr_quality_score(text: str) -> float:
    t = re.sub(r"\s+", "", text)
    if not t:
        return -1.0
    score = 0.0
    if "=" in t:
        score += 6.0
    if re.search(r"[xX]", t):
        score += 4.0
    if re.search(r"\d", t):
        score += 2.5
    if re.search(r"[²³^]", t) or re.search(r"\*\*", t):
        score += 2.0
    if re.fullmatch(r"[0-9xX+\-*/=().,a-zA-Z^²³⁰¹⁴⁵⁶⁷⁸⁹]+", t):
        score += 1.0
    score += min(len(t), 64) * 0.02
    return score


def _fix_common_math_ocr_errors(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)

    for pat, repl in [
        (r"(?i)yut", "4x+"),
        (r"(?i)yux", "4x+"),
        (r"(?i)lul", "4x+"),
        (r"(?i)lut", "4x+"),
        (r"(?i)vut", "4x+"),
    ]:
        s = re.sub(pat, repl, s)

    s = re.sub(r"(?i)y(?=[xu])", "4", s)
    s = re.sub(r"(?i)(\d)u(?=[+xX=0-])", r"\1x", s)
    s = re.sub(r"(?i)(\d)u(\d)", r"\1x+\2", s)

    s = re.sub(r"(?i)(?<=[xX])t(?=\d)", "+", s)
    s = re.sub(r"(\d)t(\d)", r"\1+\2", s)

    if "=" not in s and re.search(r"[xXuU]", s):
        s = re.sub(r"\+(\d+)-0$", r"+\1=0", s)
        if "=" not in s:
            s = re.sub(r"(\d+)-0$", r"\1=0", s)

    s = _fix_cursive_x_power_ocr(s)

    return s


def _fix_cursive_x_power_ocr(s: str) -> str:
    """
    Handwriting OCR often reads cursive x² as '02' (zero + two) and x as 'h'/'n'/'u'.
    Example: x²-4x+4=0 → 02-4h+4=0 → x**2-4x+4=0
    """
    if not s or "=" not in s:
        return s

    # Leading "02","002",… then + or - : almost always x**2 (invalid Python int "02" otherwise)
    s = re.sub(r"^0+2(?=[+-])", "x**2", s, count=1)
    s = re.sub(r"^0+3(?=[+-])", "x**3", s, count=1)

    # Same pattern after opening "(", or after "+"/"-" at start of a parenthetical chunk (rare)
    s = re.sub(r"(?<=[+\-])0+2(?=[+-])", "x**2", s)
    s = re.sub(r"(?<=[+\-])0+3(?=[+-])", "x**3", s)

    # Variable x read as h, n, u when preceded by a digit and followed by + - = or another digit
    s = re.sub(r"(?i)(?<=[0-9])([hnu])(?=[+=\-0-9]|=)", "x", s)

    # -h+ …  or +h+ …  (coefficient 1 or -1 lost, letter misread)
    s = re.sub(r"(?i)(?<=[+\-])([hnu])(?=[+=\-0-9]|=)", "x", s)
    return s


def _normalize_for_sympy(expr: str) -> str:
    """Turn OCR power notation into SymPy (**, not ^)."""
    s = expr
    unicode_powers = {
        "⁰": "**0",
        "¹": "**1",
        "²": "**2",
        "³": "**3",
        "⁴": "**4",
        "⁵": "**5",
        "⁶": "**6",
        "⁷": "**7",
        "⁸": "**8",
        "⁹": "**9",
    }
    for u, repl in unicode_powers.items():
        s = s.replace(u, repl)
    s = re.sub(r"([xX0-9\)])²", r"\1**2", s)
    s = re.sub(r"([xX0-9\)])³", r"\1**3", s)
    s = re.sub(r"\^(\d+)", r"**\1", s)
    s = s.replace("^", "**")
    return s


# OCR attempt: (kwargs for RapidOCR __call__)
_OCR_KW_VARIANTS = (
    {},
    {"text_score": 0.35, "box_thresh": 0.35},
    {"text_score": 0.2, "box_thresh": 0.25},
    {"text_score": 0.12, "box_thresh": 0.2},
)


def _collect_image_variants(base: np.ndarray, enhanced: np.ndarray) -> list[np.ndarray]:
    """Several visual pipelines so at least one triggers the detector on hard crops."""
    variants = [
        _ensure_min_size(base.copy()),
        enhanced,
        _invert_if_dark(_ensure_min_size(base.copy())),
        _autocontrast(_ensure_min_size(base.copy())),
    ]
    h, w = base.shape[:2]
    if max(h, w) < 1100:
        b2 = cv2.resize(base, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        variants.append(_enhance_for_math_ocr(b2))
    seen = set()
    out = []
    for v in variants:
        key = v.shape[:2]
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _ocr_best_text_from_path(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    base = _resize_rgb(img)
    enhanced = _enhance_for_math_ocr(base.copy())

    ocr = _get_ocr()
    best_text = ""
    best_score = (-1.0, -1)

    for arr in _collect_image_variants(base, enhanced):
        for kw in _OCR_KW_VARIANTS:
            try:
                result, _ = ocr(arr, **kw)
            except TypeError:
                result, _ = ocr(arr)
            joined = _join_ocr_tokens(result)
            fixed = _fix_common_math_ocr_errors(joined)
            if not fixed:
                continue
            sc = (_math_ocr_quality_score(fixed), len(fixed))
            if sc > best_score:
                best_score = sc
                best_text = fixed

    return best_text


def _normalize_math_text(text: str) -> str:
    t = text.replace("×", "*").replace("·", "*").replace("÷", "/")
    t = t.replace("_", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _sympy_transformations():
    return standard_transformations + (implicit_multiplication_application,)


def _build_solution(problem_text: str) -> str:
    raw = _normalize_math_text(problem_text)
    if not raw:
        return "Could not read any text from the image. Try a clearer photo."

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    main = " ".join(lines) if lines else raw
    main = re.sub(r"\s+", "", main)
    main = _normalize_for_sympy(main)

    tr = _sympy_transformations()
    local: dict[str, Any] = {"sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E}

    if "=" in main:
        parts = main.split("=", 1)
        lhs_s, rhs_s = parts[0].strip(), parts[1].strip()
        try:
            x = sp.symbols("x")
            local["x"] = x
            lhs = parse_expr(lhs_s, transformations=tr, local_dict=local)
            rhs = parse_expr(rhs_s, transformations=tr, local_dict=local)
            eq = sp.Eq(lhs, rhs)
            sol = sp.solve(eq, x)
            steps = [
                f"Extracted: {main}",
                "",
                "Symbolic setup:",
                f"  {sp.pretty(eq)}",
                "",
                f"Solution for x: {sol}",
                "",
                f"Final Answer: x = {sol}",
            ]
            return "\n".join(steps)
        except Exception as e:
            return (
                f"Extracted: {main}\n\n"
                f"SymPy could not solve this as a single equation in x.\n"
                f"Detail: {e}\n"
                "Tip: use `x**2` or `x²`, e.g. `x**2-5*x+6=0`."
            )

    try:
        expr = parse_expr(main, transformations=tr, local_dict=local)
        simplified = sp.simplify(expr)
        expanded = sp.expand(simplified)
        lines_out = [
            f"Extracted: {main}",
            "",
            "Simplified:",
            f"  {sp.pretty(simplified)}",
            "",
            "Expanded:",
            f"  {sp.pretty(expanded)}",
            "",
            f"Final Answer: {simplified}",
        ]
        return "\n".join(lines_out)
    except Exception as e:
        return (
            f"Extracted: {main}\n\n"
            f"SymPy could not parse this expression.\n"
            f"Detail: {e}\n"
            "Tip: use plain math notation (e.g. 2*x+1, x**2, sqrt(x))."
        )


def problem_label_from_text(problem_text: str) -> str:
    raw = (problem_text or "").strip()
    lines = raw.splitlines()
    line = lines[0].strip() if lines else raw
    if len(line) > 48:
        return line[:45] + "..."
    return line or "Math problem"


def solve_math_image(image_path: str) -> dict:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(image_path)

    problem_text = _ocr_best_text_from_path(str(path))
    solution = _build_solution(problem_text)
    label = problem_label_from_text(problem_text)
    return {
        "problem_statement": problem_text.strip() or "(no text detected)",
        "solution": solution,
        "problem_label": label,
    }
