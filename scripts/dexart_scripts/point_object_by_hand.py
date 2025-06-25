import cv2
import os
import csv
from utils import DATA_DIR

# Path
ROOT_DIR = DATA_DIR / "planning_datasets/bucket_dex_art_dataset/dexart_all_bucket_png/bucket_viz"
ROOT_DIR = str(ROOT_DIR)

OUTPUT_FILE = DATA_DIR / "planning_datasets/bucket_dex_art_dataset/anchor_points_bucket.csv"
OUTPUT_FILE = str(OUTPUT_FILE)

results = []

def click_event(event, x, y, flags, params):
    global clicked, clicked_x, clicked_y
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        clicked_x, clicked_y = x, y

clicked = False
clicked_x, clicked_y = 0, 0

# traverse all subfolders, find the first frame_000.png
subdirs = sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])

for subdir in subdirs:
    image_path = os.path.join(ROOT_DIR, subdir, "frame_000.png")
    if not os.path.exists(image_path):
        print(f"Missing frame_000.png in {subdir}, skipping.")
        continue

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to read image: {image_path}")
        continue

    clicked = False
    cv2.imshow("Click Anchor Point", frame)
    cv2.setMouseCallback("Click Anchor Point", click_event)

    while True:
        cv2.imshow("Click Anchor Point", frame)
        key = cv2.waitKey(1) & 0xFF
        if clicked:
            print(f"Clicked on {subdir}/frame_000.png at ({clicked_x}, {clicked_y})")
            results.append((f"{subdir}/frame_000.png", clicked_x, clicked_y))
            break
        if key == ord('q'):  # press q to skip current image
            print(f"Skipped {subdir}/frame_000.png")
            break

cv2.destroyAllWindows()

# save clicked results
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'x', 'y'])
    writer.writerows(results)

print(f"Saved anchor points to {OUTPUT_FILE}")
