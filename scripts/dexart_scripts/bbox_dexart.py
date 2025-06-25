import os
import cv2
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import glob
import supervision as sv
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.video_utils import create_video_from_images
from utils import DATA_DIR


SAM2_CKPT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
ANCHOR_CSV = DATA_DIR / "planning_datasets/bucket_dex_art_dataset/anchor_points_bucket.csv"
ANCHOR_CSV = str(ANCHOR_CSV)
INPUT_FRAME_DIR = DATA_DIR / "planning_datasets/bucket_dex_art_dataset/dexart_all_bucket_png/bucket_viz"
INPUT_FRAME_DIR = str(INPUT_FRAME_DIR)
OUTPUT_BASE_DIR = DATA_DIR / "planning_datasets/bucket_dex_art_dataset/results_bbox/bucket"
OUTPUT_BASE_DIR = str(OUTPUT_BASE_DIR)
REVERSED_FRAME_DIR = DATA_DIR / "planning_datasets/bucket_dex_art_dataset/tmp_reversed_bucket"
REVERSED_FRAME_DIR = str(REVERSED_FRAME_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_image_sequence_with_point(
    frame_dir, save_dir, output_video_path,
    video_predictor, image_predictor,
    ann_frame_idx, point_coord, object_name="object"
):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    frame_names = sorted([
        f for f in os.listdir(frame_dir) if f.endswith(".png")
    ], key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))

    if len(frame_names) == 0:
        print(f"[skip] image frames are empty: {frame_dir}")
        return

    tmp_jpg_dir = os.path.join(save_dir, "jpg_frames")
    os.makedirs(tmp_jpg_dir, exist_ok=True)

    for i, png_name in enumerate(frame_names):
        img = cv2.imread(os.path.join(frame_dir, png_name))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {png_name}")
        jpg_path = os.path.join(tmp_jpg_dir, f"{i:05d}.jpg")
        cv2.imwrite(jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    inference_state = video_predictor.init_state(video_path=tmp_jpg_dir)
    img_path = os.path.join(frame_dir, frame_names[ann_frame_idx])
    image_source = cv2.imread(img_path)
    image_predictor.set_image(image_source)

    point_coords = np.array([point_coord])
    point_labels = np.array([1])
    masks, _, _ = image_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    input_box = sv.mask_to_xyxy(masks[0:1])[0]
    object_id = 1

    video_predictor.add_new_points_or_box(
        inference_state,
        frame_idx=ann_frame_idx,
        obj_id=object_id,
        box=input_box,
        points=point_coords,
        labels=point_labels
    )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    reversed_names = list(reversed(frame_names))
    if os.path.exists(REVERSED_FRAME_DIR):
        shutil.rmtree(REVERSED_FRAME_DIR)
    Path(REVERSED_FRAME_DIR).mkdir(parents=True)

    for idx, fname in enumerate(reversed_names):
        src = os.path.join(frame_dir, fname)
        dst = os.path.join(REVERSED_FRAME_DIR, f"{idx:05d}.jpg")
        img = cv2.imread(src)
        if img is None:
            raise FileNotFoundError(f"cannot read source frame: {src}")
        cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    inference_state_rev = video_predictor.init_state(REVERSED_FRAME_DIR)
    rev_idx = len(frame_names) - 1 - ann_frame_idx
    rev_img_path = os.path.join(REVERSED_FRAME_DIR, f"{rev_idx:05d}.jpg")
    rev_img = cv2.imread(rev_img_path)
    if rev_img is None:
        raise FileNotFoundError(f"cannot read reversed image frame: {rev_img_path}")
    image_predictor.set_image(rev_img)

    video_predictor.add_new_points_or_box(
        inference_state_rev,
        frame_idx=rev_idx,
        obj_id=object_id,
        box=input_box,
        points=point_coords,
        labels=point_labels
    )

    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state_rev):
        true_frame_idx = len(frame_names) - 1 - out_frame_idx
        if true_frame_idx not in video_segments:
            video_segments[true_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    all_frame_boxes = {}
    ID_TO_OBJECTS = {1: object_name}

    for frame_idx, segments in video_segments.items():
        if frame_idx < 0 or frame_idx >= len(frame_names):
            continue

        frame_name = frame_names[frame_idx]
        img = cv2.imread(os.path.join(frame_dir, frame_name))
        object_ids = list(segments.keys())
        masks = np.concatenate(list(segments.values()), axis=0)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            class_id=np.array(object_ids, dtype=np.int32),
        )

        for i, obj_id in enumerate(object_ids):
            mask_np = (detections.mask[i] * 255).astype(np.uint8)
            mask_dir = os.path.join(save_dir, ID_TO_OBJECTS[obj_id])
            os.makedirs(mask_dir, exist_ok=True)
            cv2.imwrite(os.path.join(mask_dir, f"mask_{frame_name}"), mask_np)

        frame_box_info = [
            [ID_TO_OBJECTS[obj_id], detections.xyxy[i].tolist()]
            for i, obj_id in enumerate(object_ids)
        ]
        all_frame_boxes[frame_name] = frame_box_info

        annotated = sv.BoxAnnotator().annotate(img.copy(), detections)
        annotated = sv.MaskAnnotator().annotate(annotated, detections)
        cv2.imwrite(os.path.join(save_dir, f"annotated_{frame_name}"), annotated)

    with open(os.path.join(save_dir, "video_boxes.json"), "w") as f:
        json.dump(all_frame_boxes, f, indent=2)

    create_video_from_images(save_dir, output_video_path)
    print(f"[done] {frame_dir} → {output_video_path}")



if __name__ == "__main__":
    df = pd.read_csv(ANCHOR_CSV)

    video_predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT)
    sam2_image_model = build_sam2(SAM2_CFG, SAM2_CKPT)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    if os.path.exists(OUTPUT_BASE_DIR):
        shutil.rmtree(OUTPUT_BASE_DIR)
    os.makedirs(OUTPUT_BASE_DIR)

    for idx, row in df.iterrows():
        filename = row["filename"]
        frame_path = Path(filename)
        
        folder_name = frame_path.parent.name
        frame_name = frame_path.name

        try:
            ann_frame_idx = int(frame_name.replace("frame_", "").replace(".png", ""))
        except:
            print(f"[skip] cannot parse frame number: {frame_name}")
            continue

        x = int(row["x"])
        y = int(row["y"])

        frame_dir = os.path.join(INPUT_FRAME_DIR, folder_name)
        if not os.path.isdir(frame_dir):
            print(f"[skip] missing image frame directory: {frame_dir}")
            continue

        save_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)
        output_path = os.path.join(save_dir, "annotated_video.mp4")

        print(f"[process] {folder_name} frame {ann_frame_idx}, point=({x},{y})")
        process_image_sequence_with_point(
            frame_dir=frame_dir,
            save_dir=save_dir,
            output_video_path=output_path,
            video_predictor=video_predictor,
            image_predictor=image_predictor,
            ann_frame_idx=ann_frame_idx,
            point_coord=(x, y),
            object_name="object"
        )
