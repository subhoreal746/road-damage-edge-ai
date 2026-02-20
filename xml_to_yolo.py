import os
import xml.etree.ElementTree as ET

BASE_DIR = "RDD2022/Japan/train"  # change to Japan later

XML_DIR = os.path.join(BASE_DIR, "annotations")
LBL_DIR = os.path.join(BASE_DIR, "labels")

os.makedirs(LBL_DIR, exist_ok=True)

CLASS_MAP = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3
}

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

for xml_file in os.listdir(XML_DIR):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(XML_DIR, xml_file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    label_path = os.path.join(LBL_DIR, xml_file.replace(".xml", ".txt"))

    with open(label_path, "w") as f:
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()

            if name not in CLASS_MAP:
                continue

            cls_id = CLASS_MAP[name]
            bbox = obj.find("bndbox")

            box = (
                float(bbox.find("xmin").text),
                float(bbox.find("xmax").text),
                float(bbox.find("ymin").text),
                float(bbox.find("ymax").text),
            )

            bb = convert_bbox((w, h), box)
            f.write(f"{cls_id} {' '.join(map(str, bb))}\n")

print("✅ XML → YOLO conversion done (COLAB-COMPATIBLE)")

