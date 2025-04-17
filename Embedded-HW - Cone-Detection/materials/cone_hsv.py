import sys
import cv2 as cv
import numpy as np
import datetime
import argparse
import os

def main(argv):
    parser = argparse.ArgumentParser(description=" cone detection")
    parser.add_argument("filename", nargs="?", default="frame000106.jpg", help="Input image file or folder containing images")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration via OpenCL")
    args = parser.parse_args(argv)

    use_gpu = args.use_gpu
    print("Using GPU acceleration (OpenCL)" if use_gpu else "Using CPU processing")

    # Determine if the provided input is a file or a folder.
    input_path = args.filename
    if os.path.isdir(input_path):
        # Get all image files in the directory
        files = sorted([
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not files:
            print("No image files found in the directory:", input_path)
            sys.exit(1)
    else:
        files = [input_path]

    # Load the cone template (shared across all images)
    cone_template_img = cv.imread(cv.samples.findFile("cone_template.png"), cv.IMREAD_GRAYSCALE)
    if cone_template_img is None:
        print("Could not open or find the cone template image.")
        sys.exit(1)

    # Process each image file one by one
    for file in files:
        print(f"\nProcessing file: {file}")

        # ===== Stage: Image input =====
        t0 = datetime.datetime.now()
        input_img = cv.imread(cv.samples.findFile(file), cv.IMREAD_COLOR)
        if input_img is None:
            print("Could not open or find the image:", file)
            continue
        t_after_loading = datetime.datetime.now()
        print(f"Image input stage: {(t_after_loading - t0).total_seconds()} seconds")

        # ===== Stage: UMat conversion =====
        if use_gpu:
            input_image = cv.UMat(input_img)
            cone_template = cv.UMat(cone_template_img)
        else:
            input_image = input_img
            cone_template = cone_template_img
        t_after_umat = datetime.datetime.now()
        print(f"UMat conversion stage: {(t_after_umat - t_after_loading).total_seconds()} seconds")

        t1 = datetime.datetime.now()  # (Original overall processing start)

        # ===== Stage: HSV segmentation & Morphological processing =====
        t_before_hsv = datetime.datetime.now()
        hsv = cv.cvtColor(input_image, cv.COLOR_BGR2HSV)
        cone_colors = {
            "Blue":   ((70, 100,  50), (135, 255, 255)),
            "Orange": (( 4, 150, 100), (18, 255, 255)),
            "Yellow": ((17, 100, 110), (40, 255, 255))
        }
        blue_mask   = cv.inRange(hsv, *cone_colors["Blue"])
        orange_mask = cv.inRange(hsv, *cone_colors["Orange"])
        yellow_mask = cv.inRange(hsv, *cone_colors["Yellow"])

        color_segmented = cv.bitwise_or(cv.bitwise_or(blue_mask, orange_mask), yellow_mask)

        # Morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        color_segmented = cv.morphologyEx(color_segmented, cv.MORPH_OPEN, kernel)
        color_segmented = cv.morphologyEx(color_segmented, cv.MORPH_CLOSE, kernel)
        t_after_morph = datetime.datetime.now()
        print(f"HSV segmentation and morphological processing stage: {(t_after_morph - t_before_hsv).total_seconds()} seconds")

        #  Count non-zero after morphological ops
        nonzero_pixels = cv.countNonZero(color_segmented)
        print(f" Non-zero pixels in color_segmented after morphology: {nonzero_pixels}")

        # ===== Stage: ROI =====
        t_before_roi = datetime.datetime.now()
        roi_mask = apply_region_of_interest(color_segmented, input_image)
        if use_gpu:
            roi_mask_cpu = roi_mask.get()
            color_segmented_cpu = color_segmented.get()
        else:
            roi_mask_cpu = roi_mask
            color_segmented_cpu = color_segmented
        t_after_roi = datetime.datetime.now()
        print(f"ROI stage: {(t_after_roi - t_before_roi).total_seconds()} seconds")

        # ===== Stage: Raw Template Matching =====
        t_before_tm = datetime.datetime.now()
        rects = []
        template_threshold = 0.65
        for i in range(1, 20):
            scale_factor = 0.25 + (i / 10.0)
            new_width  = max(int(cone_template_img.shape[1] * scale_factor), 1)
            new_height = max(int(cone_template_img.shape[0] * scale_factor), 1)

            if use_gpu:
                tmp_resize = cv.resize(cone_template, (new_width, new_height))
                tmp_resize_cpu = tmp_resize.get()
            else:
                tmp_resize = cv.resize(cone_template, (new_width, new_height))
                tmp_resize_cpu = tmp_resize

            res = cv.matchTemplate(roi_mask_cpu, tmp_resize_cpu, cv.TM_CCORR_NORMED)
            loc = np.where(res >= template_threshold)
            for pt in zip(*loc[::-1]):
                rects.append((int(pt[0]), int(pt[1]), int(tmp_resize_cpu.shape[1]),
                              int(tmp_resize_cpu.shape[0]), res[pt[1], pt[0]], scale_factor))
        t_after_tm = datetime.datetime.now()
        print(f"Raw template matching stage: {(t_after_tm - t_before_tm).total_seconds()} seconds")
        print(f" Raw detections from template matching: {len(rects)}")

        # ===== Stage: Non-Maximum Suppression (NMS) =====
        t_before_nms = datetime.datetime.now()
        filtered_rects = apply_non_maximum_suppression(rects)
        t_after_nms = datetime.datetime.now()
        print(f"NMS stage: {(t_after_nms - t_before_nms).total_seconds()} seconds")
        print(f" Detections after NMS: {len(filtered_rects)}")

        # ===== Stage: Filter False Positives =====
        t_before_filter = datetime.datetime.now()
        filtered_rects = filter_false_positives(filtered_rects)
        t_after_filter = datetime.datetime.now()
        print(f"Filter false positives stage: {(t_after_filter - t_before_filter).total_seconds()} seconds")
        print(f" Detections after filter_false_positives: {len(filtered_rects)}")

        # ===== Stage: Color Labelling & Coverage Check =====
        t_before_color_label = datetime.datetime.now()
        if use_gpu:
            output_image = input_image.get().copy()
        else:
            output_image = input_image.copy()

        coverage_threshold = 0.07
        for rect in filtered_rects:
            x, y, w, h, score, scale = rect
            coverage_ratio = compute_color_coverage_ratio(color_segmented_cpu, x, y, w, h)
            print(f" BBox {x,y,w,h}, coverage={coverage_ratio:.3f}, score={score:.3f}")

            if coverage_ratio < coverage_threshold:
                print(" Skipping, coverage below threshold.")
                continue

            detected_color = get_cone_color(hsv, x, y, w, h, cone_colors)
            color_map = {"Blue": (255, 0, 0), "Orange": (0, 165, 255), "Yellow": (0, 255, 255)}
            box_color = color_map.get(detected_color, (0, 255, 0))
            cv.rectangle(output_image, (x, y), (x + w, y + h), box_color, 2)
            cv.putText(output_image, detected_color, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        t_after_color_label = datetime.datetime.now()
        print(f"Color labelling & coverage check stage: {(t_after_color_label - t_before_color_label).total_seconds()} seconds")

        t2 = datetime.datetime.now()
        print(f"Processing time: {(t2 - t1).total_seconds()} seconds")

        # Save output image (the output filename now includes the original filename)
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output_detected_cones_" + os.path.basename(file))
        cv.imwrite(output_path, output_image)
        print(f"Output image saved to {output_path}")

        cv.waitKey(0)
        cv.destroyAllWindows()

def apply_region_of_interest(mask, image):
    if hasattr(image, "get"):
        img_cpu = image.get()
    else:
        img_cpu = image
    height, width = img_cpu.shape[:2]
    roi = np.zeros((height, width), dtype=np.uint8)
    cv.rectangle(roi, (0, int(height * 0.2)), (width, height), 255, -1)
    return cv.bitwise_and(mask, roi)

def compute_color_coverage_ratio(segmented_mask, x, y, w, h):
    height, width = segmented_mask.shape[:2]
    x_end = min(x + w, width)
    y_end = min(y + h, height)
    if x_end <= x or y_end <= y:
        return 0.0
    roi = segmented_mask[y:y_end, x:x_end]
    on_pixels = cv.countNonZero(roi)
    total_pixels = roi.size
    coverage_ratio = on_pixels / float(total_pixels)
    return coverage_ratio

def apply_non_maximum_suppression(rects):
    if not rects:
        return []
    rects = sorted(rects, key=lambda x: x[4], reverse=True)
    nms_rects = []
    while rects:
        best = rects.pop(0)
        nms_rects.append(best)
        rects = [r for r in rects if not overlap(best, r)]
    return nms_rects

def overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1[:4]
    x2, y2, w2, h2 = rect2[:4]
    inter_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    inter_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter_area = inter_x * inter_y
    area1 = w1 * h1
    area2 = w2 * h2
    return (inter_area > 0.5 * min(area1, area2))

def filter_false_positives(rects):
    kept = []
    for rect in rects:
        x, y, w, h, score, scale = rect
        if w > 20 and h > 30:
            aspect_ratio = float(w)/float(h)
            if 0.5 < aspect_ratio < 2.0:
                kept.append(rect)
    return kept

def get_cone_color(hsv, x, y, w, h, cone_colors):
    if hasattr(hsv, "get"):
        hsv_cpu = hsv.get()
    else:
        hsv_cpu = hsv
    cropped = hsv_cpu[y:y+h, x:x+w]
    counts = {}
    for color, (lower, upper) in cone_colors.items():
        mask = cv.inRange(cropped, np.array(lower), np.array(upper))
        counts[color] = cv.countNonZero(mask)
    sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    best_color, best_cov = sorted_colors[0]
    second_color, second_cov = sorted_colors[1]
    diff = best_cov - second_cov
    if diff < 50:
        avg_hue = compute_average_hue(cropped)
        if avg_hue < 15:
            return "Orange"
        elif avg_hue < 65:
            return "Blue"
        else:
            return "Yellow"
    else:
        return best_color

def compute_average_hue(cropped_bgr_or_hsv):
    hsv_region = cv.cvtColor(cropped_bgr_or_hsv, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_region)
    mask = (h>0)|(s>0)|(v>0)
    nonzero_h = h[mask]
    if nonzero_h.size == 0:
        return 0
    return float(np.mean(nonzero_h))

if __name__ == "__main__":
    main(sys.argv[1:])
