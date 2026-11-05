import cv2


face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(frame):
    """Find the largest face in the frame and return it as a 250x250 image, or None if no face is found."""
    scale = 0.5
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = [int(v / scale) for v in max(faces, key=lambda f: f[2] * f[3])]
    pad = int(0.2 * w)
    y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
    x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
    face = frame[y1:y2, x1:x2]
    return cv2.resize(face, (250, 250))