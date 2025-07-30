from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import io
import time
import threading
import logging
from typing import List, Dict, Any
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pose Estimation API", version="1.0.0")

model = None
model_lock = threading.Lock()

class ImageRequest(BaseModel):
    id: str
    image: str

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    probability: float

class PoseResponse(BaseModel):
    id: str
    count: int
    boxes: List[BoundingBox]
    keypoints: List[List[List[float]]]
    speed_preprocess: float
    speed_inference: float
    speed_postprocess: float

class AnnotatedPoseResponse(BaseModel):
    id: str
    count: int
    annotated_image: str  # base64 encoded annotated image
    speed_preprocess: float
    speed_inference: float
    speed_postprocess: float

def load_model():
    global model
    with model_lock:
        if model is None:
            logger.info("Loading YOLO pose estimation model...")
            model = YOLO('/app/model3-yolol/yolo11l-pose.pt')
            logger.info("Model loaded successfully")
    return model

def decode_base64_image(base64_string: str) -> np.ndarray:
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
            
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

def encode_image_to_base64(image: np.ndarray) -> str:
    try:
        _, buffer = cv2.imencode('.jpg', image)
        
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to encode image: {str(e)}")

def draw_pose_annotations(image: np.ndarray, results) -> np.ndarray:
    try:
        annotated_image = image.copy()
        
        colors = {
            'bbox': (0, 255, 0),
            'keypoint': (0, 0, 255),
            'skeleton': (255, 0, 0)
        }
        
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), colors['bbox'], 2)
                    
                    label = f"Person: {conf:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['bbox'], 2)
            
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                keypoint_scores = result.keypoints.conf.cpu().numpy()
                
                for person_kpts, person_scores in zip(keypoints, keypoint_scores):
                    for connection in skeleton_connections:
                        kpt1_idx, kpt2_idx = connection
                        
                        if (kpt1_idx < len(person_kpts) and kpt2_idx < len(person_kpts) and
                            person_scores[kpt1_idx] > 0.5 and person_scores[kpt2_idx] > 0.5):
                            
                            x1, y1 = map(int, person_kpts[kpt1_idx])
                            x2, y2 = map(int, person_kpts[kpt2_idx])
                            
                            cv2.line(annotated_image, (x1, y1), (x2, y2), colors['skeleton'], 2)
                    
                    for i, (kpt, score) in enumerate(zip(person_kpts, person_scores)):
                        if score > 0.5:
                            x, y = map(int, kpt)
                            cv2.circle(annotated_image, (x, y), 4, colors['keypoint'], -1)
                            cv2.circle(annotated_image, (x, y), 4, (255, 255, 255), 1)
                            
                            cv2.putText(annotated_image, str(i), (x + 5, y - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return annotated_image
        
    except Exception as e:
        logger.error(f"Error drawing annotations: {str(e)}")
        return image

def process_pose_estimation(image: np.ndarray, request_id: str) -> PoseResponse:
    model = load_model()
    
    start_preprocess = time.time()
    
    preprocess_time = time.time() - start_preprocess
    
    start_inference = time.time()
    results = model(image)
    inference_time = time.time() - start_inference
    
    start_postprocess = time.time()
    
    boxes = []
    keypoints_list = []
    count = 0
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                boxes.append(BoundingBox(
                    x=float(x1),
                    y=float(y1),
                    width=float(width),
                    height=float(height),
                    probability=conf
                ))
        
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()
            
            count = len(keypoints_xy)
            
            for person_idx in range(count):
                person_keypoints = []
                for kp_idx in range(len(keypoints_xy[person_idx])):
                    x, y = keypoints_xy[person_idx][kp_idx]
                    conf = keypoints_conf[person_idx][kp_idx]
                    person_keypoints.append([float(x), float(y), float(conf)])
                keypoints_list.append(person_keypoints)
    
    postprocess_time = time.time() - start_postprocess
    
    return PoseResponse(
        id=request_id,
        count=count,
        boxes=boxes,
        keypoints=keypoints_list,
        speed_preprocess=preprocess_time,
        speed_inference=inference_time,
        speed_postprocess=postprocess_time
    )

def process_annotated_pose_estimation(image: np.ndarray, request_id: str) -> AnnotatedPoseResponse:
    model = load_model()
    
    start_preprocess = time.time()
    
    preprocess_time = time.time() - start_preprocess
    
    start_inference = time.time()
    results = model(image)
    inference_time = time.time() - start_inference
    
    start_postprocess = time.time()
    
    count = 0
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            count += len(result.boxes)
    
    annotated_image = draw_pose_annotations(image, results)
    
    annotated_image_base64 = encode_image_to_base64(annotated_image)
    
    postprocess_time = time.time() - start_postprocess
    
    return AnnotatedPoseResponse(
        id=request_id,
        count=count,
        annotated_image=annotated_image_base64,
        speed_preprocess=preprocess_time,
        speed_inference=inference_time,
        speed_postprocess=postprocess_time
    )

@app.on_event("startup")
async def startup_event():
    load_model()
    logger.info("API server started successfully")

@app.get("/")
async def root():
    return {"message": "Pose Estimation API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/pose", response_model=PoseResponse)
async def pose_estimation(request: ImageRequest):
    try:
        logger.info(f"Processing pose estimation request with ID: {request.id}")
        
        result_holder = {"response": None, "error": None}
        
        def run_pose_estimation():
            try:
                image = decode_base64_image(request.image)
                
                result_holder["response"] = process_pose_estimation(image, request.id)
            except Exception as e:
                logger.error(f"Error in background thread for {request.id}: {str(e)}")
                result_holder["error"] = str(e)
        
        thread = threading.Thread(target=run_pose_estimation)
        thread.start()
        thread.join()
        
        if result_holder["error"]:
            raise Exception(result_holder["error"])
        
        response = result_holder["response"]
        if response is None:
            raise Exception("Failed to process image in background thread")
        
        logger.info(f"Completed pose estimation for request ID: {request.id}, "
                   f"detected {response.count} person(s)")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request {request.id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/pose/annotated", response_model=AnnotatedPoseResponse)
async def annotated_pose_estimation(request: ImageRequest):
    try:
        logger.info(f"Processing annotated pose estimation request with ID: {request.id}")
        
        result_holder = {"response": None, "error": None}
        
        def run_annotated_pose_estimation():
            try:
                image = decode_base64_image(request.image)
                
                result_holder["response"] = process_annotated_pose_estimation(image, request.id)
            except Exception as e:
                logger.error(f"Error in annotated processing thread for {request.id}: {str(e)}")
                result_holder["error"] = str(e)
        
        thread = threading.Thread(target=run_annotated_pose_estimation)
        thread.start()
        thread.join()
        
        if result_holder["error"]:
            raise Exception(result_holder["error"])
        
        response = result_holder["response"]
        if response is None:
            raise Exception("Failed to process annotated image in background thread")
        
        logger.info(f"Completed annotated pose estimation for request ID: {request.id}, "
                   f"detected {response.count} person(s)")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing annotated request {request.id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


