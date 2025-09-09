#!/usr/bin/env python3
# pip install opencv-python scikit-image numpy
import argparse, json, os, math, csv
import cv2
import numpy as np
from skimage.measure import CircleModel, ransac

# ------------------------ helpers ------------------------

def parse_crop(crop_str):
    if not crop_str:
        return None
    x, y, w, h = map(int, crop_str.split(","))
    return (x, y, w, h)

def calibrate_ppmm_by_click(img, known_mm=50.0, win="Calibration"):
    """Interactive calibration: click two points on the ruler that are `known_mm` apart.
    Returns (pixels_per_mm, (x1,y1,x2,y2)) in FULL-IMAGE coordinates."""
    pts = []
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(img_disp, (x, y), 6, (255, 0, 255), -1)
            if len(pts) == 2:
                cv2.line(img_disp, pts[0], pts[1], (255, 255, 0), 2)
            cv2.imshow(win, img_disp)

    img_disp = img.copy()
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, img_disp)
    cv2.setMouseCallback(win, on_mouse)
    print(f"[calibrate] Click two points on the ruler that are exactly {known_mm} mm apart. Press ESC to cancel.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(win)
            raise RuntimeError("Calibration canceled.")
        if len(pts) == 2:
            break
    cv2.destroyWindow(win)
    p1, p2 = np.array(pts[0], float), np.array(pts[1], float)
    dist_px = float(np.linalg.norm(p1 - p2))
    return (dist_px / known_mm, (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])))

def calibrate_ppmm_from_pts(scale_pts, known_mm):
    """Non-GUI calibration from 'x1,y1,x2,y2' string. Returns (ppmm, (x1,y1,x2,y2))."""
    x1, y1, x2, y2 = map(float, scale_pts.split(","))
    dist_px = math.hypot(x2 - x1, y2 - y1)
    return (dist_px / known_mm, (x1, y1, x2, y2))

def save_scale_json(path, pixels_per_mm, known_mm=None, source_image=None, scale_points_full_px=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "pixels_per_mm": float(pixels_per_mm),
        "known_mm": float(known_mm) if known_mm is not None else None,
        "source_image": source_image,
        "scale_points_full_px": scale_points_full_px,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path

def load_scale_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if "pixels_per_mm" not in data:
        raise ValueError(f"{path} does not contain 'pixels_per_mm'.")
    return float(data["pixels_per_mm"]), data

def _score_contour(cnt, img_w, img_h, prefer_center=False, center_radius_frac=0.4):
    a = cv2.contourArea(cnt)
    p = cv2.arcLength(cnt, True)
    circ = (4.0 * math.pi * a) / (p*p + 1e-12) if p > 0 else 0.0
    score = a * circ  # large & round
    if prefer_center and a > 0:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
            img_cx, img_cy = img_w/2.0, img_h/2.0
            d = math.hypot(cx - img_cx, cy - img_cy)
            R = center_radius_frac * min(img_w, img_h)
            if d < R:
                score *= 2.0
    return score

def fit_best_circle(contour):
    pts = contour[:, 0, :].astype(float)
    model_robust, inliers = ransac(pts, CircleModel, min_samples=3,
                                   residual_threshold=2.0, max_trials=1000)
    if model_robust is None:
        model = CircleModel()
        model.estimate(pts)
        xc, yc, r = model.params
        inliers = np.ones(len(pts), dtype=bool)
    else:
        xc, yc, r = model_robust.params
    return (float(xc), float(yc), float(r)), pts, inliers

def fit_best_ellipse(contour):
    """
    Fit an ellipse to a contour (requires >=5 points).
    Returns:
      (xc, yc), (major_axis, minor_axis), angle_deg
    where axes are FULL lengths (not radii).
    """
    if len(contour) < 5:
        raise RuntimeError("Not enough points to fit an ellipse (need >=5).")
    cnt = contour.astype(np.float32)
    (xc, yc), (MA, ma), angle = cv2.fitEllipse(cnt)
    if ma > MA:  # ensure MA >= ma
        MA, ma = ma, MA
    return (float(xc), float(yc)), (float(MA), float(ma)), float(angle)

def radial_deviation_metrics(pts, xc, yc, r, mm_per_px=None):
    radii = np.sqrt((pts[:,0]-xc)**2 + (pts[:,1]-yc)**2)
    dev = radii - r  # signed deviation (px)
    rms = float(np.sqrt(np.mean(dev**2)))
    max_abs = float(np.max(np.abs(dev)))
    std = float(np.std(dev))
    rng = float(np.max(dev) - np.min(dev))  # raw difference (px)
    metrics = {
        "rms_px": rms,
        "max_abs_px": max_abs,
        "std_px": std,
        "range_px": rng,
    }
    if mm_per_px:
        metrics.update({
            "rms_mm": rms * mm_per_px,
            "max_abs_mm": max_abs * mm_per_px,
            "std_mm": std * mm_per_px,
            "range_mm": rng * mm_per_px,
        })
    return metrics

def segment_mask_or_edges(gray, args):
    """Return (binary_mask, edges). Only one will be non-None based on method."""
    if args.edge_method == "binary":
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if not args.binary_invert else cv2.THRESH_BINARY,
            args.binary_block_size, args.binary_C
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        if args.erode > 0:
            th = cv2.erode(th, kernel, iterations=args.erode)
        return th, None
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray)
        edges = cv2.Canny(gray_eq, args.canny_low, args.canny_high, L2gradient=True)
        return None, edges

def draw_scale_line(img, scale_pts_abs, color_line=(255,255,0), color_pts=(255,0,255),
                    thickness=2, radius=5, offset=(0,0)):
    """Draw calibration line/endpoints if they fall inside current image (apply crop offset)."""
    if not scale_pts_abs:
        return
    x1, y1, x2, y2 = scale_pts_abs
    x_off, y_off = offset
    p1 = (int(round(x1 - x_off)), int(round(y1 - y_off)))
    p2 = (int(round(x2 - x_off)), int(round(y2 - y_off)))
    h, w = img.shape[:2]
    def inside(p): return 0 <= p[0] < w and 0 <= p[1] < h
    if inside(p1) or inside(p2):
        cv2.line(img, p1, p2, color_line, thickness)
        cv2.circle(img, p1, radius, color_pts, -1)
        cv2.circle(img, p2, radius, color_pts, -1)

# ---- contour selection with diameter filter (uses known scale if available) ----

def pick_contour_with_filter(mask_or_edges, prefer_center, center_radius_frac,
                             mm_per_px, min_diam_mm, max_diam_mm):
    """Return the best contour that passes diameter filter (if scale available)."""
    h, w = mask_or_edges.shape[:2]
    cnts, _ = cv2.findContours(mask_or_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise RuntimeError("No contours found.")

    scored = sorted(
        cnts,
        key=lambda c: _score_contour(c, w, h, prefer_center, center_radius_frac),
        reverse=True
    )

    if mm_per_px and (min_diam_mm > 0 or max_diam_mm < 1e9):
        for c in scored:
            (xc, yc, r), pts, inliers = fit_best_circle(c)
            diam_mm = 2.0 * r * mm_per_px
            if min_diam_mm <= diam_mm <= max_diam_mm:
                return c, (xc, yc, r), pts, inliers
        c = scored[0]
        return c, *fit_best_circle(c)

    c = scored[0]
    return c, *fit_best_circle(c)

# ------------------------ core ------------------------

def analyze_image(path, args):
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_prefix = os.path.join(args.out_dir, stem)

    img_full = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_full is None:
        raise FileNotFoundError(path)

    # optional crop
    if args.crop:
        x, y, w, h = args.crop
        img = img_full[y:y+h, x:x+w].copy()
        offset = (x, y)
    else:
        img = img_full.copy()
        offset = (0, 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)

    binary, edges = segment_mask_or_edges(gray, args)

    # debug saves
    if args.save_debug:
        if binary is not None:
            cv2.imwrite(os.path.join(args.out_dir, f"{stem}_binary.png"), binary)
        if edges is not None:
            cv2.imwrite(os.path.join(args.out_dir, f"{stem}_edges.png"), edges)

    # ---- calibration priority: scale-from -> scale-pts -> click-scale -> pixels-per-mm ----
    scale_pts_abs = None
    scale_meta = None
    if args.scale_from:
        ppmm, scale_meta = load_scale_json(args.scale_from)
    elif args.scale_pts:
        ppmm, scale_pts_abs = calibrate_ppmm_from_pts(args.scale_pts, args.known_mm)
    elif args.click_scale:
        try:
            ppmm, scale_pts_abs = calibrate_ppmm_by_click(img_full.copy(), known_mm=args.known_mm)
        except cv2.error:
            raise SystemExit("OpenCV GUI not available. Use --scale-pts or --pixels-per-mm.")
    else:
        ppmm = args.pixels_per_mm if args.pixels_per_mm else None

    mm_per_px = (1.0 / ppmm) if ppmm else None

    # ---- choose contour (with diameter filter) ----
    if args.edge_method == "binary":
        cnt, (xc, yc, r), pts, inliers = pick_contour_with_filter(
            binary, args.prefer_center, args.center_radius_frac,
            mm_per_px, args.min_diameter_mm, args.max_diameter_mm
        )
    else:
        try:
            cnt, (xc, yc, r), pts, inliers = pick_contour_with_filter(
                edges, args.prefer_center, args.center_radius_frac,
                mm_per_px, args.min_diameter_mm, args.max_diameter_mm
            )
        except RuntimeError:
            # Hough fallback → build a synthetic contour
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_eq = clahe.apply(gray)
            circles = cv2.HoughCircles(
                gray_eq, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(gray.shape)//3,
                param1=max(50, args.canny_high), param2=30,
                minRadius=int(0.15*min(gray.shape)), maxRadius=int(0.6*min(gray.shape))
            )
            if circles is None:
                raise RuntimeError("Canny gave no contour and HoughCircles found nothing.")
            circles = np.uint16(np.around(circles))
            best = max(circles[0,:], key=lambda c: c[2])
            xc0, yc0, r0 = map(float, best)

            if mm_per_px:
                diam_mm0 = 2.0 * r0 * mm_per_px
                if not (args.min_diameter_mm <= diam_mm0 <= args.max_diameter_mm):
                    raise RuntimeError(f"Hough circle {diam_mm0:.2f} mm out of bounds.")

            if edges is None:
                edges = cv2.Canny(gray_eq, max(10, args.canny_low//2), max(40, args.canny_high//2), L2gradient=True)
            ys, xs = np.where(edges > 0)
            dr = np.sqrt((xs - xc0)**2 + (ys - yc0)**2) - r0
            mask = np.abs(dr) < 3.0
            pts = np.stack([xs[mask], ys[mask]], axis=1).astype(np.float32)
            if len(pts) < 20:
                raise RuntimeError("Hough found a circle but too few rim edge points nearby.")
            cnt = pts.reshape(-1,1,2).astype(np.int32)
            (xc, yc, r), pts, inliers = fit_best_circle(cnt)

    # metrics from circle
    area = float(cv2.contourArea(cnt))
    per  = float(cv2.arcLength(cnt, True))
    circularity = (4.0 * math.pi * area) / (per*per + 1e-12)
    diameter_px = 2.0 * r
    diameter_mm = (diameter_px * mm_per_px) if mm_per_px else None

    # final diameter filter (safety)
    if diameter_mm is not None:
        if not (args.min_diameter_mm <= diameter_mm <= args.max_diameter_mm):
            raise RuntimeError(f"Detected rim {diameter_mm:.2f} mm is outside "
                               f"[{args.min_diameter_mm}, {args.max_diameter_mm}] mm")

    # out-of-round stats
    dev = radial_deviation_metrics(pts, xc, yc, r, mm_per_px)

    # optional ellipse fit
    ellipse = None
    if args.fit_ellipse:
        try:
            (xec, yec), (MA, ma), angle = fit_best_ellipse(cnt)
            ellipse = {
                "center_px": (xec, yec),
                "major_px": MA,
                "minor_px": ma,
                "angle_deg": angle,
                "ellipticity": (MA / ma) if ma > 0 else None,  # >= 1
                "eccentricity": math.sqrt(1.0 - (ma/MA)**2) if MA > 0 else None,
                "area_px2": math.pi * (MA/2.0) * (ma/2.0),
                "ovalization": (MA - ma) / ((MA + ma)/2.0) if (MA + ma) > 0 else None
            }
            if mm_per_px:
                ellipse.update({
                    "major_mm": MA * mm_per_px,
                    "minor_mm": ma * mm_per_px,
                    "area_mm2": ellipse["area_px2"] * (mm_per_px**2),
                    "ovalization_mm": (MA - ma) * mm_per_px
                })
        except Exception as e:
            ellipse = {"error": str(e)}

    # export points (Cartesian + cylindrical)
    x0, y0 = xc, yc
    dx = pts[:,0] - x0
    dy = pts[:,1] - y0
    r_vec = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    theta_deg = np.degrees(theta)

    rows = []
    for i in range(len(pts)):
        x_px = float(pts[i,0]); y_px = float(pts[i,1])
        row = {
            "x_px": x_px + offset[0],
            "y_px": y_px + offset[1],
            "theta_rad": float(theta[i]),
            "theta_deg": float(theta_deg[i]),
            "r_px": float(r_vec[i]),
            "x_center_px": x0 + offset[0],
            "y_center_px": y0 + offset[1],
        }
        if mm_per_px:
            row.update({
                "x_mm": (x_px + offset[0]) * mm_per_px,
                "y_mm": (y_px + offset[1]) * mm_per_px,
                "r_mm": float(r_vec[i] * mm_per_px),
            })
        rows.append(row)

    csv_path = f"{out_prefix}_points.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # overlay (single image)
    overlay_path = None
    if not args.no_overlay:
        dbg = img.copy()
        # contour
        if args.edge_method == "binary":
            cv2.drawContours(dbg, [cnt], -1, (255, 0, 0), 1)      # blue = contour
        else:
            cv2.drawContours(dbg, [cnt], -1, (0, 165, 255), 1)    # orange = contour
        # raw points
        for (x_i, y_i) in pts.astype(int):
            cv2.circle(dbg, (x_i, y_i), 1, (0, 0, 255), -1)       # red dots
        # fitted circle + center
        cv2.circle(dbg, (int(round(xc)), int(round(yc))), int(round(r)), (0, 255, 0), 2)  # green
        cv2.circle(dbg, (int(round(xc)), int(round(yc))), 3, (0, 255, 255), -1)           # center
        # optional ellipse (purple)
        if ellipse and "major_px" in ellipse:
            cv2.ellipse(
                dbg,
                (int(round(ellipse["center_px"][0])), int(round(ellipse["center_px"][1]))),
                (int(round(ellipse["major_px"]/2.0)), int(round(ellipse["minor_px"]/2.0))),
                ellipse["angle_deg"], 0, 360, (200, 0, 200), 2
            )

        # draw scale if provided/loaded (only if inside crop)
        scale_pts_abs = None
        if args.scale_from:
            _, scale_meta = load_scale_json(args.scale_from)
            scale_pts_abs = (scale_meta or {}).get("scale_points_full_px")
        elif args.scale_pts:
            _, scale_pts_abs = calibrate_ppmm_from_pts(args.scale_pts, args.known_mm)
        draw_scale_line(dbg, scale_pts_abs, offset=offset)

        label = f"Diam: {diameter_px:.0f}px"
        if diameter_mm is not None:
            label = f"Diam: {diameter_mm:.2f} mm"
        if ellipse and ellipse.get("ellipticity"):
            label += f"  e={ellipse['ellipticity']:.3f}"
        cv2.putText(dbg, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(dbg, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        overlay_path = f"{out_prefix}_overlay.png"
        cv2.imwrite(overlay_path, dbg)

    # metrics JSON
    metrics = {
        "image": os.path.abspath(path),
        "pixels_per_mm": ppmm,
        "known_mm_for_scale": args.known_mm if (args.click_scale or args.scale_pts) else (scale_meta or {}).get("known_mm") if args.scale_from else None,
        "center_px": [float(x0 + offset[0]), float(y0 + offset[1])],
        "radius_px": float(r),
        "diameter_px": float(diameter_px),
        "diameter_mm": float(diameter_mm) if diameter_mm is not None else None,
        "area_px2": float(area),
        "perimeter_px": float(per),
        "circularity_4piA_P2": float(circularity),
        "rms_out_of_round_px": dev["rms_px"],
        "max_out_of_round_px": dev["max_abs_px"],
        "std_out_of_round_px": dev["std_px"],
        "range_out_of_round_px": dev["range_px"],
        "inlier_ratio": float(np.mean(inliers)),
        "ellipse": ellipse,
        "csv_points": os.path.abspath(csv_path),
        "overlay_image": os.path.abspath(overlay_path) if overlay_path else None
    }
    if mm_per_px:
        metrics.update({
            "center_mm": [(x0 + offset[0]) * mm_per_px, (y0 + offset[1]) * mm_per_px],
            "radius_mm": r * mm_per_px,
            "perimeter_mm": per * mm_per_px,
            "area_mm2": area * (mm_per_px**2),
            "rms_out_of_round_mm": dev["rms_mm"],
            "max_out_of_round_mm": dev["max_abs_mm"],
            "std_out_of_round_mm": dev["std_mm"],
            "range_out_of_round_mm": dev["range_mm"],
        })

    with open(f"{out_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if args.save_scale and ppmm:
        save_scale_json(args.save_scale, ppmm,
                        known_mm=metrics.get("known_mm_for_scale"),
                        source_image=os.path.abspath(path),
                        scale_points_full_px=None)

    return metrics

# ------------------------ CLI ------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="Detect a circular can, filter by diameter (mm), export rim points/metrics, optional ellipse fit, draw scale, save/load pixel-to-mm."
    )
    p.add_argument("image", help="Path to JPEG/PNG image.")
    p.add_argument("--out-dir", default="out", help="Output directory (default: ./out).")

    # Calibration
    p.add_argument("--pixels-per-mm", type=float, default=None, help="Known pixels-per-mm.")
    p.add_argument("--click-scale", action="store_true", help="Click 2 ruler points to calibrate.")
    p.add_argument("--known-mm", type=float, default=50.0, help="Distance in mm for the 2 clicked points.")
    p.add_argument("--scale-pts", type=str, default=None, help="Non-GUI scale 'x1,y1,x2,y2' (full image).")
    p.add_argument("--save-scale", type=str, default=None, help="Write JSON with {'pixels_per_mm': ...}.")
    p.add_argument("--scale-from", type=str, default=None, help="Load JSON scale (reused across images).")

    # Diameter filter (mm)
    p.add_argument("--min-diameter-mm", type=float, default=0.0, help="Ignore rims smaller than this (mm).")
    p.add_argument("--max-diameter-mm", type=float, default=1e9, help="Ignore rims larger than this (mm).")

    # Segmentation
    p.add_argument("--edge-method", choices=["binary", "canny"], default="binary", help="Rim detection method.")
    p.add_argument("--binary-block-size", type=int, default=51, help="Adaptive threshold block size (odd).")
    p.add_argument("--binary-C", type=int, default=2, help="Adaptive threshold constant C.")
    p.add_argument("--binary-invert", action="store_true", help="Invert binary (if needed).")
    p.add_argument("--erode", type=int, default=0, help="Erode iterations (contract mask).")
    p.add_argument("--canny-low", type=int, default=75, help="Canny low threshold.")
    p.add_argument("--canny-high", type=int, default=200, help="Canny high threshold.")

    # Selection bias & debug
    p.add_argument("--prefer-center", action="store_true", help="Prefer contours near image center.")
    p.add_argument("--center-radius-frac", type=float, default=0.4, help="Center preference radius fraction.")
    p.add_argument("--save-debug", action="store_true", help="Save *_edges.png / *_binary.png.")
    p.add_argument("--crop", type=str, default=None, help="Crop ROI 'x,y,w,h'.")
    p.add_argument("--no-overlay", action="store_true", help="Skip overlay PNG.")

    # Ellipse
    p.add_argument("--fit-ellipse", action="store_true",
                   help="Also fit/draw a best-fit ellipse and report ellipse metrics.")
    return p

def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.crop = parse_crop(args.crop)
    metrics = analyze_image(args.image, args)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
