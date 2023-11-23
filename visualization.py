import glob
import json
import os.path

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import pickle

top_model = pickle.load(open("models/top.pkl", "rb"))
bottom_model = pickle.load(open("models/bottom.pkl", "rb"))


def calc_size(x1, x0, text: str):
    num_upper = len([t for t in text if t.isupper()])
    # num_digit = len([t for t in text if t.isdigit()])
    length = len(text) + 0.4 * num_upper
    return int(1.5 * (x1 - x0) / length + 2 * np.log1p(len(text)))


def visualization(path, blocks):
    image = Image.open(path)
    blank = Image.new(mode="RGB", size=image.size, color="white")
    draw = ImageDraw.Draw(blank)
    heights = sorted(block["y1"] - block["y0"] for block in blocks if block.get('type') == "word")
    if len(heights) >= 5:
        avg_height = np.mean(heights[int(0.3 * len(heights)): int(0.7 * len(heights))])
    elif len(heights) >= 1:
        avg_height = heights[len(heights) // 2]
    else:
        avg_height = 0
    for block in blocks:

        if block.get('type') == "word" and len(block["text"]) > 0:
            size = int(avg_height) + 1
            draw.text((block["x0"] + size // 3, block["y1"] - size), block['text'], fill="blue",
                      font=ImageFont.truetype("arial.ttf", size=size), stroke_width=0)
    new_image = Image.new("RGB", size=(image.width * 2, image.height), color="white")
    new_image.paste(image, (0, 0))
    new_image.paste(blank, (image.width, 0))
    return new_image


if __name__ == '__main__':
    engine = "capture_screen/compression"
    image_path = f"data/capture_screen/compression"
    output_path = f"data/visualization/{engine}"
    os.makedirs(output_path, exist_ok=True)
    for json_path in tqdm(glob.glob(f"data/output/{engine}/*.json")):
        name = os.path.basename(json_path)[:-5]
        img_path = os.path.join(image_path, name + ".jpeg")
        image = visualization(img_path, json.load(open(json_path, encoding="utf-8")))
        image.save(f"{output_path}/{name}.jpg")
