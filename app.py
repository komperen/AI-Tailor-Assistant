import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
import base64
from flask import Flask, request, jsonify

# =========================
# APP INIT
# =========================
app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

pose = mp_pose.Pose(model_complexity=2)
holistic = mp_holistic.Holistic()

DEFAULT_HEIGHT_CM = 152.0
DEPTH_MODEL_NAME = "DPT_Hybrid"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================
# CAMERA PROFILE
# =========================
CAMERA_TYPE = "phone"  # options: webcam | phone

CAMERA_CONFIG = {
    "webcam": {
        "FOCAL_LENGTH": 900,
        "IDEAL_DISTANCE": 180
    },
    "phone": {
        "FOCAL_LENGTH": 1250,
        "IDEAL_DISTANCE": 220
    }
}

FOCAL_LENGTH = CAMERA_CONFIG[CAMERA_TYPE]["FOCAL_LENGTH"]
IDEAL_DISTANCE_CM = CAMERA_CONFIG[CAMERA_TYPE]["IDEAL_DISTANCE"]


# =========================
# LOAD DEPTH MODEL
# =========================
def load_depth_model():
    print(f"Loading depth model: {DEPTH_MODEL_NAME}")
    model = torch.hub.load("intel-isl/MiDaS", DEPTH_MODEL_NAME)
    model.eval().to(device)
    return model

depth_model = load_depth_model()

# =========================
# DEPTH ESTIMATION
# =========================
def estimate_depth(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
    tensor = torch.tensor(rgb, dtype=torch.float32).permute(2,0,1).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=(384,384),
        mode="bilinear",
        align_corners=False
    ).to(device)

    with torch.no_grad():
        depth = depth_model(tensor)

    return depth.squeeze().cpu().numpy()

# =========================
# SCALE FROM HEIGHT
# =========================
def corrected_scale(pixel_height, real_height):

    raw_scale = real_height / pixel_height

    distance = estimate_camera_distance(pixel_height, real_height)

    correction = distance / IDEAL_DISTANCE_CM

    return raw_scale * correction

def calculate_scale(landmarks, img_h, height_cm):

    top = landmarks[mp_pose.PoseLandmark.NOSE.value].y * img_h
    bottom = max(
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    ) * img_h

    pixel_height = abs(bottom - top)

    return corrected_scale(pixel_height, height_cm)


def estimate_camera_distance(pixel_height, real_height):

    return (real_height * FOCAL_LENGTH) / pixel_height



# =========================
# BODY WIDTH DETECTOR
# =========================
def get_width(frame, y, cx_ratio):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _,th = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)

    y = int(np.clip(y,0,frame.shape[0]-1))
    row = th[y]

    cx = int(cx_ratio * frame.shape[1])
    left = right = cx

    for i in range(cx,0,-1):
        if row[i]==0:
            left=i
            break

    for i in range(cx,len(row)):
        if row[i]==0:
            right=i
            break

    return max(right-left, frame.shape[1]*0.1)
# =========================
# CALCULATE CHEST 
# =========================

def detect_body_type(shoulder_px, hip_px):

    ratio = shoulder_px / max(hip_px, 1e-6)

    if ratio > 1.25:
        return "V", 0.82
    elif ratio > 1.10:
        return "Athletic", 0.78
    elif ratio > 0.95:
        return "Normal", 0.75
    else:
        return "Wide", 0.72


def sample_depth_patch(depth_map, x, y, w, h, patch=5):

    dh, dw = depth_map.shape

    sx = int(x * dw / w)
    sy = int(y * dh / h)

    x1 = max(sx - patch, 0)
    x2 = min(sx + patch + 1, dw)
    y1 = max(sy - patch, 0)
    y2 = min(sy + patch + 1, dh)

    patch_vals = depth_map[y1:y2, x1:x2]

    if patch_vals.size == 0:
        return float(depth_map[sy, sx])

    return float(np.median(patch_vals))

# =========================
# MEASUREMENT CORE
# =========================

def calculate_measurements(results, frame, scale):

    lm = results.pose_landmarks.landmark
    h,w,_ = frame.shape

    depth_map = estimate_depth(frame)

    measurements = {}

    # ================= SHOULDER
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    shoulder_px = abs(ls.x-rs.x)*w
    measurements["shoulder_width"] = round(shoulder_px*scale,2)

    # ================= CHEST
    chest,cy,cx,width_px,body_type = calculate_chest(
        frame,lm,scale,depth_map
    )

    measurements["chest_circumference"] = chest
    measurements["body_type"] = body_type

    left = int(cx-width_px/2)
    right = int(cx+width_px/2)

    cv2.line(frame,(left,cy),(right,cy),(0,0,255),4)

    cv2.putText(frame,
        f"Chest: {chest} cm",
        (left,cy-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,0,255),
        2)

    cv2.putText(frame,
        f"Type: {body_type}",
        (left,cy+20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,0,255),
        2)

    axis=(int(width_px/2),int(width_px*0.35))
    cv2.ellipse(frame,(cx,cy),axis,0,0,360,(0,0,255),2)

    # ================= SHIRT LENGTH
    shirt_px = abs(
        lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y -
        lm[mp_pose.PoseLandmark.LEFT_HIP.value].y
    )*h

    measurements["shirt_length"]=round(shirt_px*scale*1.15,2)

    # ================= SLEEVE
    sleeve_px = abs(
        lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y -
        lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    )*h

    measurements["sleeve_length"]=round(sleeve_px*scale*1.05,2)

    return measurements,frame

# =========================
# CHEST DEPTH CALC
# =========================
def calculate_chest(frame,lm,scale,depth_map):

    h,w,_ = frame.shape

    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # ================= WIDTH FROM POSE
    shoulder_px = abs(ls.x-rs.x)*w
    hip_px = abs(lh.x-rh.x)*w

    body_type,ratio = detect_body_type(shoulder_px,hip_px)

    # calculate chest width from shoulder ratio
    chest_width_px = shoulder_px * ratio

    # safety clamp supaya tidak lebih besar dari torso realistis
    chest_width_px = min(chest_width_px, shoulder_px*0.85)

    # ================= CHEST POSITION
    chest_y = ls.y + (lh.y-ls.y)*0.18
    y_px = int(chest_y*h)

    center_x = int((ls.x+rs.x)/2*w)

    # ================= DEPTH SAMPLE
    depth_val = sample_depth_patch(depth_map, center_x, y_px, w, h)

    dmin = float(np.min(depth_map))
    dmax = float(np.max(depth_map))
    norm = (depth_val-dmin)/max(dmax-dmin,1e-6)

    depth_ratio = 0.55 + (1-norm)*0.35

    # ================= CM
    width_cm = chest_width_px*scale
    depth_cm = width_cm*depth_ratio

    a = width_cm/2
    b = depth_cm/2

    raw_chest = 2*np.pi*np.sqrt((a*a+b*b)/2)
    offset = width_cm * 0.065
    chest = raw_chest - offset

    return round(chest,2),y_px,center_x,chest_width_px,body_type


# =========================
# VALIDATION
# =========================
def validate(img):
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res=holistic.process(rgb)
    return res.pose_landmarks is not None,res

# =========================
# API ENDPOINT
# =========================
@app.route("/upload_images",methods=["POST"])
def upload():

    if "front" not in request.files:
        return jsonify({"error":"Front image required"}),400

    height=float(request.form.get("height_cm",DEFAULT_HEIGHT_CM))

    frame=cv2.imdecode(
        np.frombuffer(request.files["front"].read(),np.uint8),
        cv2.IMREAD_COLOR
    )

    valid,res=validate(frame)
    if not valid:
        return jsonify({"error":"Pose not detected"}),400

    scale=calculate_scale(res.pose_landmarks.landmark,frame.shape[0],height)

    measurements,annotated=calculate_measurements(res,frame,scale)

    _,buf=cv2.imencode(".jpg",annotated)
    img64=base64.b64encode(buf).decode()

    return jsonify({
        "measurements":measurements,
        "annotated_image":img64,
        "scale":round(scale,6)
    })

# =========================
# RUN
# =========================
if __name__=="__main__":
    print("Server running on :8001")
    app.run(host="0.0.0.0",port=8001)