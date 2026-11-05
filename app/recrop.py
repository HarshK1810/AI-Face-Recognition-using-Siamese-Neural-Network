import os
import cv2
from face_utils import crop_face

folders = {
    os.path.join('data', 'anchor'): os.path.join('data_face', 'anchor'),
    os.path.join('data', 'positive'): os.path.join('data_face', 'positive'),
    os.path.join('data', 'negative'): os.path.join('data_face', 'negative'),
    os.path.join('data', 'negative_webcam'): os.path.join('data_face', 'negative_webcam'),
}

for src, dst in folders.items():
    os.makedirs(dst, exist_ok=True)
    kept, skipped = 0, 0
    for name in os.listdir(src):
        img = cv2.imread(os.path.join(src, name))
        if img is None:
            skipped += 1
            continue
        face = crop_face(img)
        if face is None:
            skipped += 1
            continue
        cv2.imwrite(os.path.join(dst, name), face)
        kept += 1
    print(f"{src}: kept {kept}, skipped {skipped} (no face found)")