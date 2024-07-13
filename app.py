from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List
import json
from PIL import Image
import io
import base64
from dress_mask import get_mask_location

app = FastAPI()

class PoseKeypoints(BaseModel):
    pose_keypoints_2d: List[List[float]]

@app.post("/process_pose_keypoints/")
async def process_pose_keypoints(
    data: str = Form(...),
    model_parse: UploadFile = File(...),
    tops: str = Form(...),  # New string parameter
):
    pose_keypoints = PoseKeypoints.parse_raw(data)
    image_content = await model_parse.read()
    
    # Load the image from the uploaded file
    image = Image.open(io.BytesIO(image_content))
    
    mask, mask_gray = get_mask_location('dc', 'dresses', image, pose_keypoints.dict())
    # Create mask and mask_gray images (dummy images for this example)
    #mask = Image.new('L', image.size, color=128)  # Dummy mask image
    #mask_gray = Image.new('L', image.size, color=200)  # Dummy mask_gray image
    
    # Convert images to bytes
    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format='PNG')
    mask_bytes = mask_bytes.getvalue()
    
    mask_gray_bytes = io.BytesIO()
    mask_gray.save(mask_gray_bytes, format='PNG')
    mask_gray_bytes = mask_gray_bytes.getvalue()
    
    # Encode bytes to base64
    mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')
    mask_gray_base64 = base64.b64encode(mask_gray_bytes).decode('utf-8')
    
    return {
        "received_data": pose_keypoints.dict(),
        "mask": mask_base64,
        "mask_gray": mask_gray_base64,
        "tops": tops  # Include the new parameter in the response
    }
