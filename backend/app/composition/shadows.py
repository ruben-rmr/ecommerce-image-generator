"""
Shadow synthesis: contact, drop, fake ambient occlusion, scene-directional shadow.

All shadows are rendered as a single-channel "shadow alpha" (HxW float32 in [0..1])
that the caller multiplies onto the canvas with a chosen color/opacity.
"""

import cv2
import numpy as np


def _alpha_binary(alpha: np.ndarray, threshold: int = 16) -> np.ndarray:
    return (alpha > threshold).astype(np.uint8) * 255


def _vertical_fade_ramp(shape: tuple[int, int], fade: float) -> np.ndarray:
    """
    Vertical opacity ramp in OBJECT-LOCAL coordinates: 1.0 on the bottom row
    (base, where the object touches the ground) and (1 - fade) on the top row
    (the future tip of the shadow). fade in [0..1]: 0 = no attenuation,
    1 = fully diluted tip.
    """
    oh, ow = shape
    fade = float(np.clip(fade, 0.0, 1.0))
    col = np.linspace(1.0 - fade, 1.0, oh, dtype=np.float32).reshape(-1, 1)
    return np.repeat(col, ow, axis=1)


def contact_shadow(canvas_size: tuple[int, int], alpha: np.ndarray, top_left: tuple[int, int],
                   intensity: float = 0.55, sigma: float = 3.0, band_ratio: float = 0.08) -> np.ndarray:
    """
    Sharp dark contact shadow under the object's footprint.

    canvas_size = (W, H). The shadow is rendered into a full-canvas float32 mask
    so the caller can multiply directly. `band_ratio` limits the shadow to a
    horizontal strip immediately under the object's lowest pixels.
    """
    W, H = canvas_size
    out = np.zeros((H, W), dtype=np.float32)

    binary = _alpha_binary(alpha)
    if not binary.any():
        return out

    # Erode lightly so the contact shadow is tighter than the silhouette.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary, k, iterations=1)
    blurred = cv2.GaussianBlur(eroded, (0, 0), sigmaX=sigma, sigmaY=sigma)

    oh, ow = alpha.shape[:2]
    x, y = top_left
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x2 <= x1 or y2 <= y1:
        return out

    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    crop = blurred[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0

    # Restrict to vertical band near the bottom of the object's footprint.
    ys, _ = np.where(alpha > 16)
    if ys.size:
        y_bottom_obj = int(ys.max())
        band_h = max(int(oh * band_ratio), 8)
        # Build a feathered band mask in object-local coordinates.
        band = np.zeros_like(alpha, dtype=np.float32)
        y_band_top = max(0, y_bottom_obj - int(band_h * 0.4))
        y_band_bot = min(oh, y_bottom_obj + band_h)
        band[y_band_top:y_band_bot] = 1.0
        band = cv2.GaussianBlur(band, (0, 0), sigmaX=band_h * 0.4, sigmaY=band_h * 0.4)
        crop = crop * band[sy1:sy2, sx1:sx2]

    out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], crop * intensity)
    return out


def drop_shadow(canvas_size: tuple[int, int], alpha: np.ndarray, top_left: tuple[int, int],
                squash: float = 0.18, shear: float = 0.0,
                offset: tuple[int, int] = (0, 0),
                sigma: float = 18.0, intensity: float = 0.30) -> np.ndarray:
    """
    Soft drop shadow obtained by an affine warp of the silhouette (vertical
    squash + optional horizontal shear) and a large gaussian blur.
    """
    W, H = canvas_size
    out = np.zeros((H, W), dtype=np.float32)
    binary = _alpha_binary(alpha).astype(np.float32) / 255.0
    if not binary.any():
        return out

    oh, ow = alpha.shape[:2]
    x, y = top_left

    # Anchor warp to the bottom of the silhouette so it stays grounded.
    ys, _ = np.where(alpha > 16)
    y_anchor_local = int(ys.max()) if ys.size else oh - 1

    M = np.float32([
        [1.0, shear, -shear * y_anchor_local],
        [0.0, squash, y_anchor_local * (1.0 - squash)],
    ])
    warped = cv2.warpAffine(binary, M, (ow, oh), flags=cv2.INTER_LINEAR, borderValue=0.0)
    warped = cv2.GaussianBlur(warped, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Place into canvas with offset.
    dx, dy = offset
    x1, y1 = max(0, x + dx), max(0, y + dy)
    x2, y2 = min(W, x + dx + ow), min(H, y + dy + oh)
    if x2 <= x1 or y2 <= y1:
        return out
    sx1, sy1 = x1 - (x + dx), y1 - (y + dy)
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], warped[sy1:sy2, sx1:sx2] * intensity)
    return out


def cast_shadow(canvas_size: tuple[int, int], alpha: np.ndarray, top_left: tuple[int, int],
                light_dir: tuple[float, float] = (-0.6, -0.6),
                length: float = 0.55,
                squash: float = 0.25,
                fade: float = 0.5,
                sigma_contact: float = 5.0,
                sigma_tip: float = 24.0,
                intensity: float = 0.55) -> np.ndarray:
    """
    Procedural cast shadow: foreshortening (vertical squash) + directional shear
    away from the light, with a base->tip opacity fade and a graduated penumbra
    (sharp at the contact, soft at the tip).

    Returns an HxW float32 [0..1] shadow mask (caller multiplies it onto the canvas).
    The warp writes straight into the full canvas (translation baked into the affine
    matrix), so the shadow is only ever clipped by the real canvas edge — never by the
    object's own bounding box, even when the object is scaled up. The fade and the
    penumbra ramp are warped together with the silhouette, so the projected tip is
    always the faded/soft end regardless of object shape (works universally).
    """
    W, H = canvas_size
    out = np.zeros((H, W), dtype=np.float32)
    binary = _alpha_binary(alpha).astype(np.float32) / 255.0
    if not binary.any():
        return out

    oh, ow = alpha.shape[:2]
    x, y = top_left

    # Anclamos el warp al pixel más bajo del objeto para que la sombra quede pegada al suelo.
    ys, _ = np.where(alpha > 16)
    y_anchor_local = int(ys.max()) if ys.size else oh - 1

    # Gradiente de atenuación aplicado en espacio local del objeto, ANTES del warp.
    field = binary * _vertical_fade_ramp((oh, ow), fade)

    # Rampa de distancia al contacto (0 en la base, 1 en la punta) para graduar la penumbra.
    tip = np.linspace(1.0, 0.0, oh, dtype=np.float32).reshape(-1, 1)
    tip = np.repeat(tip, ow, axis=1)

    # Vector de luz -> shear en sentido OPUESTO a la dirección de la luz.
    lx, ly = float(light_dir[0]), float(light_dir[1])
    norm = max(np.hypot(lx, ly), 1e-6)
    lx, ly = lx / norm, ly / norm
    shear_x = -lx * length

    # Traslación (x, y) horneada en la matriz: la sombra se proyecta en coordenadas de
    # canvas, así solo se recorta en el borde real del lienzo, nunca en la caja del objeto.
    M = np.float32([
        [1.0, shear_x, x - shear_x * y_anchor_local],
        [0.0, squash,  y + y_anchor_local * (1.0 - squash)],
    ])

    # Bounding box de la sombra proyectada en el canvas (transformando las 4 esquinas de la
    # caja local) + margen para el kernel del blur. Trabajar solo en esa región mantiene la
    # latencia baja aunque el warp/blur sea a coordenadas de canvas.
    corners = np.array([[0, 0], [ow, 0], [0, oh], [ow, oh]], dtype=np.float32)
    cx = M[0, 0] * corners[:, 0] + M[0, 1] * corners[:, 1] + M[0, 2]
    cy = M[1, 0] * corners[:, 0] + M[1, 1] * corners[:, 1] + M[1, 2]
    pad = int(np.ceil(3.0 * max(sigma_contact, sigma_tip))) + 2
    bx1 = max(0, int(np.floor(cx.min())) - pad)
    by1 = max(0, int(np.floor(cy.min())) - pad)
    bx2 = min(W, int(np.ceil(cx.max())) + pad)
    by2 = min(H, int(np.ceil(cy.max())) + pad)
    if bx2 <= bx1 or by2 <= by1:
        return out
    bw, bh = bx2 - bx1, by2 - by1

    # Desplazamos la traslación al origen de la bbox para warpear a un buffer reducido.
    Mc = M.copy()
    Mc[0, 2] -= bx1
    Mc[1, 2] -= by1

    # El fade y la rampa de penumbra viajan junto con la silueta: la "cabeza" del objeto
    # se convierte en la punta diluida y difusa de la sombra tras el escorzo.
    warped = cv2.warpAffine(field, Mc, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0.0)
    warped_tip = cv2.warpAffine(tip, Mc, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0.0)

    # Penumbra graduada: nítida en el contacto, difusa hacia la punta.
    sharp = cv2.GaussianBlur(warped, (0, 0), sigmaX=sigma_contact, sigmaY=sigma_contact)
    soft = cv2.GaussianBlur(warped, (0, 0), sigmaX=sigma_tip, sigmaY=sigma_tip)
    w = np.clip(warped_tip, 0.0, 1.0)
    result = sharp * (1.0 - w) + soft * w

    out[by1:by2, bx1:bx2] = result * intensity
    return out


def fake_ambient_occlusion(canvas_size: tuple[int, int], alpha: np.ndarray, top_left: tuple[int, int],
                           radius: int = 30, intensity: float = 0.15) -> np.ndarray:
    """
    Soft darkening immediately around the silhouette, simulating ambient
    occlusion. Built from the inverted distance transform OUTSIDE the silhouette.
    """
    W, H = canvas_size
    out = np.zeros((H, W), dtype=np.float32)
    binary = _alpha_binary(alpha)
    if not binary.any():
        return out

    inv = cv2.bitwise_not(binary)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    ao = np.clip(1.0 - dist / float(radius), 0.0, 1.0)
    ao = cv2.GaussianBlur(ao, (0, 0), sigmaX=radius * 0.25, sigmaY=radius * 0.25)

    oh, ow = alpha.shape[:2]
    x, y = top_left
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x2 <= x1 or y2 <= y1:
        return out
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], ao[sy1:sy2, sx1:sx2] * intensity)
    return out


def scene_shadow(canvas_size: tuple[int, int], alpha: np.ndarray, top_left: tuple[int, int],
                 light_dir: tuple[float, float],
                 length: float = 0.55,
                 squash: float = 0.28,
                 sigma: float = 22.0,
                 intensity: float = 0.40) -> np.ndarray:
    """
    Directional drop shadow using the light vector. The shadow is projected
    OPPOSITE to the light direction; squashed vertically. `length` scales the
    horizontal projection length relative to the object height.
    """
    W, H = canvas_size
    out = np.zeros((H, W), dtype=np.float32)
    binary = _alpha_binary(alpha).astype(np.float32) / 255.0
    if not binary.any():
        return out

    oh, ow = alpha.shape[:2]
    x, y = top_left
    ys, _ = np.where(alpha > 16)
    y_anchor_local = int(ys.max()) if ys.size else oh - 1

    lx, ly = float(light_dir[0]), float(light_dir[1])
    norm = max(np.hypot(lx, ly), 1e-6)
    lx, ly = lx / norm, ly / norm

    # Shear the shadow toward (-lx) and squash vertically.
    shear_x = -lx * length
    M = np.float32([
        [1.0, shear_x, -shear_x * y_anchor_local],
        [0.0, squash, y_anchor_local * (1.0 - squash)],
    ])
    warped = cv2.warpAffine(binary, M, (ow, oh), flags=cv2.INTER_LINEAR, borderValue=0.0)
    warped = cv2.GaussianBlur(warped, (0, 0), sigmaX=sigma, sigmaY=sigma)

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x2 <= x1 or y2 <= y1:
        return out
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], warped[sy1:sy2, sx1:sx2] * intensity)
    return out
