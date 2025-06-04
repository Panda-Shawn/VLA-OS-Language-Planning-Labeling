import json

import numpy as np
from PIL import ImageDraw, ImageFont
import re
import cv2 as cv
import matplotlib


def show_box(box, ax, text, color):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        matplotlib.patches.FancyBboxPatch((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2, label="hehe")
    )
    ax.text(x0, y0 + 10, f"{text}", color="white")


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def decode_instance_names(instance_names):
    instance_strings = []
    for instance in instance_names:
        name = instance.decode("utf-8")
        if name in [
            "OnTheGroundPanda0",
            "NullMount0",
            "PandaGripper0",
            "MountedPanda0",
            "RethinkMount0",
        ]:
            continue
        else:
            string = re.sub(r"(_\d+)+$", "", name)
            string = re.sub(r"_\d+_", " ", string)
            string = string.replace("_", " ")
            instance_strings.append(string)
    return instance_strings


def post_process_caption(caption, lang_instruction):
    text = caption.replace(",", ".")
    if text[-1] != ".":
        text += "."
    return text


def split_instance_masks(instance_mask, instance_id_to_names):
    instance_ids = instance_id_to_names.keys()
    binary_masks = []
    for instance_id in instance_ids:
        binary_mask = (instance_mask == instance_id).astype(np.uint8)
        binary_masks.append(binary_mask)
    return binary_masks, instance_ids


def split_gripper_mask(instance_mask, instance_id_names):
    for key, item in instance_id_names.items():
        if item == "PandaGripper0":
            gripper_mask = (instance_mask == key).astype(np.uint8)
            return gripper_mask
    raise IndexError("No Gripper Found in the Libero Image Segmentation Mask")


def compute_iou(bbox, mask):
    x1, y1, x2, y2 = map(int, bbox)
    y1, y2 = max(0, y1), min(mask.shape[0], y2)
    x1, x2 = max(0, x1), min(mask.shape[1], x2)
    if x1 >= x2 or y1 >= y2:
        return 0.0

    mask_region = mask[y1:y2, x1:x2]
    intersection = np.sum(mask_region)
    mask_area = np.sum(mask)
    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area == 0:
        return 0.0
    union = mask_area + bbox_area - intersection

    return intersection / union if union != 0 else 0.0


def remove_redundant_bbox(bboxes):
    filtered_bbox = {}
    for score, label, box in bboxes:
        if label not in filtered_bbox.keys():
            filtered_bbox[label] = (score, label, box)
        else:
            if score > filtered_bbox[label][0]:
                filtered_bbox[label] = (score, label, box)
    return [value for value in filtered_bbox.values()]


def remove_low_iou_bbox(bboxes, mask, instance_id_to_names, iou_threshold=0.5):
    binary_masks, instance_ids = split_instance_masks(mask, instance_id_to_names)
    filtered_bboxes = {}
    for score, _, bbox in bboxes:
        result = []
        for binary_mask, instance_id in zip(binary_masks, instance_ids):
            iou = compute_iou(bbox, binary_mask)
            if iou >= iou_threshold and instance_id in instance_id_to_names.keys():
                instance_name = instance_id_to_names[instance_id]
                result.append((instance_id, instance_name, iou))
        if len(result) == 1:
            instance_id, instance_name, iou = result[0]
            if instance_id not in filtered_bboxes.keys():
                filtered_bboxes[instance_id] = (score, instance_name, bbox, iou)
            else:
                filtered_bboxes[instance_id] = (
                    (score, instance_name, bbox, iou)
                    if iou > filtered_bboxes[instance_id][3]
                    else filtered_bboxes[instance_id]
                )
    return [value[:3] for value in filtered_bboxes.values()]


def mask_to_bboxes(mask, instance_id_to_names):
    binary_masks, instance_ids = split_instance_masks(mask, instance_id_to_names)
    bboxes = []
    for binary_mask, instance_id in zip(binary_masks, instance_ids):
        # Convert to binary mask
        binary_mask = (binary_mask > 0).astype(np.uint8)

        # Remove noise using morphological opening
        kernel = np.ones((3,3), np.uint8)
        binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel)

        # Find contours and use the largest one
        contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)
            box = (x, y, x + w, y + h)
            bboxes.append((instance_id_to_names[instance_id], box))
    return bboxes


def draw_gripper_position(image, gripper_position_center, radius=5):
    color = (0, 0, 255)
    thickness = -1
    cv.circle(image, gripper_position_center, radius, color, thickness)
    return image


def draw_bounding_boxes(image, bboxes, save_path):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    for label, bbox in bboxes:
        box = [int(b) for b in bbox]
        draw.rectangle(box, outline="red", width=1)
        label_text = label
        text_bbox = font.getbbox(label_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = (box[0], box[1] - text_height)
        draw.rectangle(
            [
                text_position[0],
                text_position[1],
                text_position[0] + text_width,
                text_position[1] + text_height,
            ],
            fill="red",
        )
        draw.text(text_position, label_text, fill="white", font=font)
    image.save(save_path)
