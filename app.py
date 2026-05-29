import streamlit as st
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_dilation, maximum_filter
from scipy.signal import convolve2d

st.set_page_config(layout="wide", page_title="Wara Netra")

st.markdown("""
<style>
    html, body, #root, .main {
        color: #111827 !important;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    .pipeline-card {
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        padding: 0.75rem;
        background-color: #ffffff;
        flex: 1;
        color: #111827;
    }
    .pipeline-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #f3f4f6;
        padding-bottom: 0.4rem;
    }
    .score-card {
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        padding: 1rem 1.5rem;
        background-color: #ffffff;
        text-align: center;
        color: #111827;
    }
    .score-card h3 {
        margin: 0;
        font-size: 0.8rem;
        color: #6b7280;
        font-weight: 500;
    }
    .score-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.25rem 0;
        color: #374151;
    }
    .status-pass {
        color: #059669 !important;
    }
    .status-fail {
        color: #dc2626 !important;
    }
    .status-neutral {
        color: #374151 !important;
    }
    div[data-testid="stImage"] img {
        border-radius: 0.375rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] hr {
        margin: 1rem 0;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    .app-header {
        margin-bottom: 0.25rem;
        color: #111827;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: -0.25rem;
    }
    .stCaption {
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="app-header">Wara Netra</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Iris Occlusion and Quality Detection Pipeline</p>', unsafe_allow_html=True)
st.divider()

def _img_as_float(img):
    return img.astype(np.float64) / 255.0

def _img_as_ubyte(img):
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def _draw_rect(img, r_start, r_end, c_start, c_end, value=255, width=2):
    out = img.copy()
    for w in range(width):
        r = r_start + w
        if 0 <= r < out.shape[0]:
            out[r, c_start:c_end] = value
        r = r_end - 1 - w
        if 0 <= r < out.shape[0]:
            out[r, c_start:c_end] = value
    for w in range(width):
        c = c_start + w
        if 0 <= c < out.shape[1]:
            out[r_start:r_end, c] = value
        c = c_end - 1 - w
        if 0 <= c < out.shape[1]:
            out[r_start:r_end, c] = value
    return out

def _disk(radius):
    r = int(radius)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y) <= r*r

def _draw_line(r0, c0, r1, c1):
    rr, cc = [], []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        rr.append(r)
        cc.append(c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)

def _canny(image, sigma=1.0, low_threshold=0.1, high_threshold=0.3):
    blurred = gaussian_filter(image, sigma=sigma)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = convolve2d(blurred, Kx, mode='same', boundary='symm')
    gy = convolve2d(blurred, Ky, mode='same', boundary='symm')

    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx) * 180.0 / np.pi
    direction[direction < 0] += 180.0

    rows, cols = image.shape
    suppressed = np.zeros_like(magnitude)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = direction[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                n1 = magnitude[i, j+1]
                n2 = magnitude[i, j-1]
            elif 22.5 <= angle < 67.5:
                n1 = magnitude[i+1, j-1]
                n2 = magnitude[i-1, j+1]
            elif 67.5 <= angle < 112.5:
                n1 = magnitude[i+1, j]
                n2 = magnitude[i-1, j]
            else:
                n1 = magnitude[i-1, j-1]
                n2 = magnitude[i+1, j+1]
            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                suppressed[i, j] = magnitude[i, j]

    strong = suppressed > high_threshold
    weak = (suppressed >= low_threshold) & (suppressed <= high_threshold)

    edges = np.zeros_like(suppressed, dtype=bool)
    strong_coords = list(zip(*np.where(strong)))
    stack = list(strong_coords)
    visited = np.zeros_like(suppressed, dtype=bool)

    while stack:
        r, c = stack.pop()
        if visited[r, c]:
            continue
        visited[r, c] = True
        edges[r, c] = True
        rmin = max(0, r - 1)
        rmax = min(rows - 1, r + 1)
        cmin = max(0, c - 1)
        cmax = min(cols - 1, c + 1)
        for nr in range(rmin, rmax + 1):
            for nc in range(cmin, cmax + 1):
                if weak[nr, nc] and not visited[nr, nc]:
                    stack.append((nr, nc))

    return edges

def _hough_lines(edges, threshold=30, line_length=50, line_gap=10):
    rows, cols = edges.shape
    num_thetas = max(rows, cols)
    thetas = np.linspace(-np.pi / 2, np.pi / 2, num_thetas)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    diag = int(np.sqrt(rows * rows + cols * cols))
    rhos = np.arange(-diag, diag + 1)

    edge_pts = np.argwhere(edges)
    if len(edge_pts) < 10:
        return None

    acc = np.zeros((len(rhos), num_thetas), dtype=np.int64)

    for y, x in edge_pts:
        rho_vals = x * cos_t + y * sin_t
        rho_idxs = np.round(rho_vals + diag).astype(np.intp)
        valid = (rho_idxs >= 0) & (rho_idxs < len(rhos))
        np.add.at(acc, (rho_idxs[valid], np.where(valid)[0]), 1)

    local_max = (acc >= threshold) & (acc == maximum_filter(acc, size=3))

    cos_t_vals = cos_t[np.newaxis, :]
    sin_t_vals = sin_t[np.newaxis, :]
    edge_cols = edge_pts[:, 1, np.newaxis]
    edge_rows = edge_pts[:, 0, np.newaxis]

    lines = []
    for rho_idx, theta_idx in zip(*np.where(local_max)):
        rho = rhos[rho_idx]
        c = cos_t_vals[0, theta_idx]
        s = sin_t_vals[0, theta_idx]

        d = np.abs(edge_cols * c + edge_rows * s - rho).ravel()
        near = d <= 1.0
        near_pts = edge_pts[near]
        if len(near_pts) < line_length:
            continue

        proj = -edge_cols[near] * s + edge_rows[near] * c
        order = np.argsort(proj.ravel())
        sorted_pts = near_pts[order]
        sorted_proj = proj.ravel()[order]

        gaps = sorted_proj[1:] - sorted_proj[:-1]
        breaks = np.where(gaps > line_gap)[0]

        start = 0
        for bi in breaks:
            end = int(bi) + 1
            if end - start >= line_length:
                seg = sorted_pts[start:end]
                lines.append(((int(seg[0, 1]), int(seg[0, 0])),
                              (int(seg[-1, 1]), int(seg[-1, 0]))))
            start = end
        if len(sorted_pts) - start >= line_length:
            seg = sorted_pts[start:]
            lines.append(((int(seg[0, 1]), int(seg[0, 0])),
                          (int(seg[-1, 1]), int(seg[-1, 0]))))

    return lines if lines else None

def _roi_slice(shape, scale=0.55):
    h, w = shape[:2]
    size = int(min(h, w) * scale)
    cy, cx = h // 2, w // 2
    half = size // 2
    r_start = max(0, cy - half)
    r_end = min(h, cy + half + size % 2)
    c_start = max(0, cx - half)
    c_end = min(w, cx + half + size % 2)
    return slice(r_start, r_end), slice(c_start, c_end)

def process_pipeline(image_pil, glare_threshold, kernel_size, hough_threshold):
    img_gray = image_pil.convert("L")
    img_array = np.array(img_gray, dtype=np.uint8)
    img_float = _img_as_float(img_array)

    glare_mask = img_float > glare_threshold
    selem = _disk(kernel_size)
    glare_mask_dilated = binary_dilation(glare_mask, structure=selem)

    edges = _canny(img_float, sigma=1.0, low_threshold=0.1, high_threshold=0.3)
    lines = _hough_lines(edges, threshold=hough_threshold, line_length=50, line_gap=10)

    frame_mask = np.zeros(img_array.shape, dtype=bool)
    if lines is not None:
        for p1, p2 in lines:
            rr, cc = _draw_line(p1[1], p1[0], p2[1], p2[0])
            valid = (rr >= 0) & (rr < frame_mask.shape[0]) & (cc >= 0) & (cc < frame_mask.shape[1])
            frame_mask[rr[valid], cc[valid]] = True

    noise_mask = glare_mask_dilated | frame_mask

    roi_y, roi_x = _roi_slice(img_array.shape)
    noise_roi = noise_mask[roi_y, roi_x]
    roi_occlusion_pct = float(np.sum(noise_roi)) / float(noise_roi.size) * 100.0
    is_rejected = roi_occlusion_pct > 15.0

    return {
        "grayscale": img_array,
        "glare_mask": glare_mask_dilated,
        "frame_mask": frame_mask,
        "noise_mask": noise_mask,
        "roi_y": roi_y,
        "roi_x": roi_x,
        "roi_pixels": noise_roi.size,
        "roi_occluded": int(np.sum(noise_roi)),
        "occlusion_pct": roi_occlusion_pct,
        "is_rejected": is_rejected,
    }

st.sidebar.markdown("### Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload image", type=["jpg", "jpeg", "png", "bmp", "tiff"]
)

st.sidebar.divider()

glare_threshold = st.sidebar.slider(
    "Glare Threshold",
    min_value=0.80,
    max_value=1.00,
    value=0.92,
    step=0.01,
    help="Normalized pixel intensity threshold for specular reflection isolation.",
)

kernel_size = st.sidebar.slider(
    "Morphological Kernel Size",
    min_value=1,
    max_value=15,
    value=5,
    step=2,
    help="Radius of the disk structuring element for dilation.",
)

hough_threshold = st.sidebar.slider(
    "Hough Line Threshold",
    min_value=10,
    max_value=150,
    value=30,
    step=1,
    help="Accumulator threshold for probabilistic Hough line detection.",
)

if uploaded_file is None:
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.info("Upload an iris image to run the occlusion detection pipeline.", icon="ℹ️")
else:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Failed to load image. The file may be corrupted or in an unsupported format.")
        st.stop()
    results = process_pipeline(pil_image, glare_threshold, kernel_size, hough_threshold)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-title">Stage 1 — Input</div>', unsafe_allow_html=True)
        roi_overlay = _draw_rect(
            results["grayscale"],
            results["roi_y"].start, results["roi_y"].stop,
            results["roi_x"].start, results["roi_x"].stop,
            value=220, width=2,
        )
        st.image(roi_overlay, width='stretch', clamp=True, channels="L")
        roi_size = results["roi_x"].stop - results["roi_x"].start
        st.caption(f"Grayscale | ROI: {roi_size}\u00d7{roi_size} (centered)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-title">Stage 2 — Glare Detection</div>', unsafe_allow_html=True)
        glare_disp = _img_as_ubyte(results["glare_mask"])
        st.image(glare_disp, width='stretch', clamp=True, channels="L")
        glare_count = int(np.sum(results["glare_mask"]))
        st.caption(f"Glare pixels: {glare_count}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-title">Stage 3 — Frame Detection</div>', unsafe_allow_html=True)
        frame_disp = _img_as_ubyte(results["frame_mask"])
        st.image(frame_disp, width='stretch', clamp=True, channels="L")
        frame_count = int(np.sum(results["frame_mask"]))
        st.caption(f"Line pixels: {frame_count}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-title">Stage 4 — Noise Mask</div>', unsafe_allow_html=True)
        noise_disp = _img_as_ubyte(results["noise_mask"])
        st.image(noise_disp, width='stretch', clamp=True, channels="L")
        noise_count = int(np.sum(results["noise_mask"]))
        st.caption(f"Occluded pixels: {noise_count}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    score_cols = st.columns([1, 1, 1, 1, 2])
    with score_cols[0]:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown("<h3>ROI Pixels</h3>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="score-value status-neutral">{results["roi_pixels"]:,}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{results['grayscale'].shape[0]}\u00d7{results['grayscale'].shape[1]} full frame")
        st.markdown('</div>', unsafe_allow_html=True)

    with score_cols[1]:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown("<h3>Occluded (ROI)</h3>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="score-value status-neutral">{results["roi_occluded"]:,}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{noise_count:,} in full frame")
        st.markdown('</div>', unsafe_allow_html=True)

    with score_cols[2]:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown("<h3>Occlusion Score</h3>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="score-value status-neutral">{results["occlusion_pct"]:.2f}%</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with score_cols[3]:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown("<h3>15% Threshold</h3>", unsafe_allow_html=True)
        exceeded = results["occlusion_pct"] > 15.0
        cls = "status-fail" if exceeded else "status-pass"
        label = "Exceeded" if exceeded else "Within"
        st.markdown(
            f'<div class="score-value {cls}">{label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with score_cols[4]:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown("<h3>Biometric Classification</h3>", unsafe_allow_html=True)
        if results["is_rejected"]:
            st.markdown(
                '<div class="score-value status-fail">Rejected for Biometrics</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="score-value status-pass">Accepted for Biometrics</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    occluded_pil = pil_image.copy()
    occluded_np = np.array(occluded_pil.convert("L"), dtype=np.uint8)
    occluded_np[100:200, 250:450] = 255
    occluded_np[280:380, 250:450] = 255
    occluded_np[150:350, 180:220] = 255
    occluded_np[150:350, 420:460] = 255
    occluded_pil = Image.fromarray(occluded_np)
    occluded_result = process_pipeline(
        occluded_pil, glare_threshold, kernel_size, hough_threshold
    )

    st.markdown(
        '<div style="display:flex;gap:1.5rem;justify-content:center;margin:0.5rem 0 1rem 0;'
        'padding:0.5rem;border:1px solid #d1d5db;border-radius:0.5rem;background:#ffffff;">'
        f'<span>Clean: <strong class="{"status-pass" if not results["is_rejected"] else "status-fail"}">'
        f'{"Accepted" if not results["is_rejected"] else "Rejected"}'
        f'</strong> ({results["occlusion_pct"]:.1f}%)</span>'
        f'<span style="color:#9ca3af;">|</span>'
        f'<span>Occluded: <strong class="{"status-pass" if not occluded_result["is_rejected"] else "status-fail"}">'
        f'{"Accepted" if not occluded_result["is_rejected"] else "Rejected"}'
        f'</strong> ({occluded_result["occlusion_pct"]:.1f}%)</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Occluded (Simulated) — Comparison")

    ocol1, ocol2, ocol3, ocol4 = st.columns(4)

    with ocol1:
        st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-title">Stage 1 — Input</div>', unsafe_allow_html=True)
        st.image(occluded_result["grayscale"], width='stretch', clamp=True, channels="L")
        st.caption("Grayscale with simulated glare + frame")
        st.markdown('</div>', unsafe_allow_html=True)

    with ocol2:
        st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-title">Stage 2 — Glare Detection</div>', unsafe_allow_html=True)
        og_disp = _img_as_ubyte(occluded_result["glare_mask"])
        st.image(og_disp, width='stretch', clamp=True, channels="L")
        og_count = int(np.sum(occluded_result["glare_mask"]))
        st.caption(f"Glare pixels: {og_count}")
        st.markdown('</div>', unsafe_allow_html=True)

    with ocol3:
        st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-title">Stage 3 — Frame Detection</div>', unsafe_allow_html=True)
        of_disp = _img_as_ubyte(occluded_result["frame_mask"])
        st.image(of_disp, width='stretch', clamp=True, channels="L")
        of_count = int(np.sum(occluded_result["frame_mask"]))
        st.caption(f"Line pixels: {of_count}")
        st.markdown('</div>', unsafe_allow_html=True)

    with ocol4:
        st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-title">Stage 4 — Noise Mask</div>', unsafe_allow_html=True)
        on_disp = _img_as_ubyte(occluded_result["noise_mask"])
        st.image(on_disp, width='stretch', clamp=True, channels="L")
        on_count = int(np.sum(occluded_result["noise_mask"]))
        st.caption(f"Occluded pixels: {on_count}")
        st.markdown('</div>', unsafe_allow_html=True)

    oscore_cols = st.columns([1, 1, 1, 1, 2])
    with oscore_cols[0]:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown("<h3>ROI Pixels</h3>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="score-value status-neutral">{occluded_result["roi_pixels"]:,}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{occluded_result['grayscale'].shape[0]}\u00d7{occluded_result['grayscale'].shape[1]} full frame")
        st.markdown('</div>', unsafe_allow_html=True)

    with oscore_cols[1]:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown("<h3>Occluded (ROI)</h3>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="score-value status-neutral">{occluded_result["roi_occluded"]:,}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{on_count:,} in full frame")
        st.markdown('</div>', unsafe_allow_html=True)

    with oscore_cols[2]:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown("<h3>Occlusion Score</h3>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="score-value status-neutral">{occluded_result["occlusion_pct"]:.2f}%</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with oscore_cols[3]:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown("<h3>15% Threshold</h3>", unsafe_allow_html=True)
        o_exceeded = occluded_result["occlusion_pct"] > 15.0
        o_cls = "status-fail" if o_exceeded else "status-pass"
        o_label = "Exceeded" if o_exceeded else "Within"
        st.markdown(
            f'<div class="score-value {o_cls}">{o_label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with oscore_cols[4]:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown("<h3>Biometric Classification</h3>", unsafe_allow_html=True)
        if occluded_result["is_rejected"]:
            st.markdown(
                '<div class="score-value status-fail">Rejected for Biometrics</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="score-value status-pass">Accepted for Biometrics</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Pipeline Report")

    roi_h = results["roi_y"].stop - results["roi_y"].start
    roi_w = results["roi_x"].stop - results["roi_x"].start
    full_size = f"{results['grayscale'].shape[1]}\u00d7{results['grayscale'].shape[0]}"
    lines_out = [
        "=== Wara Netra Pipeline Report ===",
        f"Image size: {full_size}",
        f"ROI: {roi_w}\u00d7{roi_h} (centered)",
        f"",
        f"--- Clean ---",
        f"  Glare pixels: {int(results['glare_mask'].sum())}",
        f"  Frame pixels: {int(results['frame_mask'].sum())}",
        f"  ROI occluded: {results['roi_occluded']} / {results['roi_pixels']}",
        f"  Occlusion: {results['occlusion_pct']:.2f}%",
        f"  Verdict: {'Accepted' if not results['is_rejected'] else 'Rejected'}",
        f"",
        f"--- Occluded (Simulated) ---",
        f"  Glare pixels: {int(occluded_result['glare_mask'].sum())}",
        f"  Frame pixels: {int(occluded_result['frame_mask'].sum())}",
        f"  ROI occluded: {occluded_result['roi_occluded']} / {occluded_result['roi_pixels']}",
        f"  Occlusion: {occluded_result['occlusion_pct']:.2f}%",
        f"  Verdict: {'Accepted' if not occluded_result['is_rejected'] else 'Rejected'}",
        f"",
        f"=== Settings ===",
        f"  Glare threshold: {glare_threshold:.2f}",
        f"  Kernel size: {kernel_size}",
        f"  Hough threshold: {hough_threshold}",
        f"  Rejection threshold: 15%",
    ]
    st.text("\n".join(lines_out))
