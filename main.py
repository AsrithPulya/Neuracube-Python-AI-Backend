import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
import re
import json
import json_repair
import random
import datetime
import asyncio
from pydantic import BaseModel
from pathlib import Path

# Load environment variables from .env file
load_dotenv()
from anthropic import AsyncAnthropic

app = FastAPI(title="Neuracube AI Orchestration Layer")

from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"Global exception occurred: {exc}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": tb}
    )

# Allow CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to your Next.js domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDGEMMA_API_URL = os.getenv("MEDGEMMA_API_URL", "https://dr7.ai/api/v1/analyze")
MEDGEMMA_API_KEY = os.getenv("MEDGEMMA_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

MEDGEMMA_CHAT_API_URL = "https://dr7.ai/api/v1/medical/chat/completions"

async def call_medgemma_text(prompt: str, system_prompt: str = "") -> str:
    if not MEDGEMMA_API_KEY:
        raise HTTPException(status_code=500, detail="MedGemma API key is not configured")
    
    headers = {
        "Authorization": f"Bearer {MEDGEMMA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": "medgemma-27b-it",
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0.2
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        res = await client.post(MEDGEMMA_CHAT_API_URL, json=data, headers=headers)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

async def call_medgemma_vision(prompt: str, base64_image: str) -> str:
    if not MEDGEMMA_API_KEY:
        raise HTTPException(status_code=500, detail="MedGemma API key is not configured")
        
    headers = {
        "Authorization": f"Bearer {MEDGEMMA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llava-med",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 8192,
        "temperature": 0.2
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        res = await client.post(MEDGEMMA_CHAT_API_URL, json=data, headers=headers)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

class ClinicalAnalysisRequest(BaseModel):
    patient_data: dict
    provider: Optional[str] = "claude"

@app.post("/api/ai/clinical-analysis")
async def analyze_clinical_scores(request: ClinicalAnalysisRequest):
    
    prompt = f"""You are a medical AI assistant.
Analyze the following patient data:
{json.dumps(request.patient_data, indent=2)}

Calculate or estimate the following clinical scores if enough data is available: ODI, NDI, DASH, QuickDASH, ASES, Constant, UCLA Shoulder, SPADI, KOOS, IKDC, Lysholm, WOMAC, Oxford Knee, HOOS, Harris Hip, Oxford Hip, AOFAS, FAAM, PRWE. 
If data is insufficient to compute or estimate a particular score, output exactly "Data not sufficient" for its value.
Also, identify any clinical "red flags" present in the primary complaint or patient history.
For the red flags, you should output TWO things:
1. `standardRedFlags`: An array of EXACT strings from this specific list: ["Fever", "Weight loss", "Cancer", "Infection", "Bowel/bladder dysfunction", "Saddle anesthesia", "Progressive neurological deficit"] if they are present.
2. `redFlags`: An array of detailed descriptions for any red flags found (both standard and other).

Return your response as a valid JSON object with the following structure:
{{
  "scores": {{
    "ODI": "value",
    "NDI": "value",
    "DASH": "value"
    // ... include all requested scores
  }},
  "standardRedFlags": [
    "Fever"
  ],
  "redFlags": [
    "Patient presents with a fever of 102F...",
    "Other red flag detail"
  ]
}}
Ensure the output is ONLY valid JSON, without any markdown formatting or surrounding text.
"""
    try:
        if request.provider == "medgemma":
            content = await call_medgemma_text(
                prompt=prompt,
                system_prompt="You are an expert orthopedic and neurosurgical AI assistant. You must output raw JSON only."
            )
        else:
            if not anthropic_client:
                raise HTTPException(status_code=500, detail="Anthropic API key is not configured")
            response = await anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                temperature=0.2,
                system="You are an expert orthopedic and neurosurgical AI assistant. You must output raw JSON only.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.content[0].text
        # extract JSON if there's backticks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        return json_repair.loads(content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/orchestrate")
async def orchestrate_analysis(
    input_data: str = Form(...),
    algorithm_data: Optional[str] = Form(None),
    dicom_files: List[UploadFile] = File(default=[])
):
    """
    Receives input data, algorithm data, and DICOM files from the frontend,
    and forwards them to the MedGemma endpoint at dr7.ai.
    """
    if not MEDGEMMA_API_KEY:
        raise HTTPException(status_code=500, detail="MedGemma API key is not configured")

    import base64
    import io
    import pydicom
    from PIL import Image
    import numpy as np

    content_text = input_data
    if algorithm_data:
        content_text += f"\nAlgorithm Context: {algorithm_data}"
        
    if dicom_files and len(dicom_files) > 0:
        content = [
            {
                "type": "text",
                "text": content_text
            }
        ]
        valid_files_added = 0
        for file in dicom_files:
            file_bytes = await file.read()
            if len(file_bytes) < 100:
                content[0]["text"] += f"\n[Warning: Skipped dummy file '{file.filename}' because it is not a valid image]"
                continue
                
            try:
                # 1. Read DICOM file from bytes
                dicom_io = io.BytesIO(file_bytes)
                ds = pydicom.dcmread(dicom_io)
                
                # 2. Extract pixel array and normalize to 0-255
                pixel_array = ds.pixel_array
                
                # Process frames in the volume (limit to max 16 to avoid API limits)
                if len(pixel_array.shape) > 2:
                    total_slices = pixel_array.shape[0]
                    max_slices = 16
                    if total_slices > max_slices:
                        indices = [int(i * total_slices / max_slices) for i in range(max_slices)]
                        slices_to_process = [pixel_array[idx] for idx in indices]
                    else:
                        slices_to_process = [pixel_array[idx] for idx in range(total_slices)]
                else:
                    slices_to_process = [pixel_array]
                
                for slice_arr in slices_to_process:
                    image_2d = slice_arr.astype(float)
                    # Avoid division by zero if image is completely black
                    if image_2d.max() > 0:
                        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
                    else:
                        image_2d_scaled = image_2d
                    image_2d_scaled = np.uint8(image_2d_scaled)
                    
                    # 3. Convert to JPEG and optimize for performance
                    image = Image.fromarray(image_2d_scaled)
                    
                    # Resize if too large (512px is optimal for FAPDIS classification)
                    if max(image.size) > 512:
                        image.thumbnail((512, 512))
                    
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG", quality=75)
                    
                    # 4. Base64 encode the JPEG
                    base64_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_encoded}"
                        }
                    })
                    valid_files_added += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                content[0]["text"] += f"\n[Warning: Failed to parse DICOM '{file.filename}': {str(e)}]"
            
        if valid_files_added == 0:
            content = content[0]["text"] # Revert back to pure text if no valid images
    else:
        content = content_text

    # Prepare the JSON data to send
    data_to_send = {
        "model": "chexagent",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {MEDGEMMA_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        # Forward the request to MedGemma using json= instead of data=
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                MEDGEMMA_API_URL,
                json=data_to_send,
                headers=headers
            )
        
        # Raise exception if the external API failed
        response.raise_for_status()
        
        # Return the exact JSON response back to the frontend
        return response.json()

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"MedGemma Error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

import json
import shutil
import time
import re
from pathlib import Path
from fastapi.responses import FileResponse

import tempfile
SESSIONS_DIR = Path(tempfile.gettempdir()) / "ohif-sessions"
TTL_SECONDS = 3600  # 1 hour
MAX_SESSIONS = 80

def prune_python_sessions():
    if not SESSIONS_DIR.exists():
        return
    try:
        sessions = []
        for p in SESSIONS_DIR.iterdir():
            if p.is_dir():
                meta_file = p / "meta.json"
                created_at = None
                if meta_file.exists():
                    try:
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                            created_at = meta.get("createdAt", None)
                    except Exception:
                        pass
                if created_at is None:
                    try:
                        created_at = int(p.stat().st_mtime * 1000)
                    except Exception:
                        created_at = int(time.time() * 1000)
                sessions.append((p, created_at))
        
        now = time.time()
        active_sessions = []
        for p, created_at in sessions:
            if now - (created_at / 1000.0) > TTL_SECONDS:
                try:
                    shutil.rmtree(p)
                except Exception:
                    pass
            else:
                active_sessions.append((p, created_at))
        
        if len(active_sessions) > MAX_SESSIONS:
            active_sessions.sort(key=lambda x: x[1])
            while len(active_sessions) > MAX_SESSIONS:
                p, _ = active_sessions.pop(0)
                try:
                    shutil.rmtree(p)
                except Exception:
                    pass
    except Exception as e:
        print(f"Error pruning sessions: {e}")

import urllib.parse

def get_tag_str(ds, name):
    val = ds.get(name, "")
    if val is None:
        return ""
    return str(val).strip().replace("\x00", "")

def get_tag_int(ds, name):
    val = ds.get(name, None)
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        return None

def get_tag_float(ds, name):
    val = ds.get(name, None)
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None

def get_tag_list(ds, name):
    val = ds.get(name, None)
    if val is None:
        return None
    try:
        if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
            return [float(x) for x in val]
        s_val = str(val).strip()
        if "\\" in s_val:
            return [float(x) for x in s_val.split("\\") if x]
        return [float(s_val)]
    except Exception:
        return None

def get_instance_metadata(ds):
    rows = get_tag_int(ds, "Rows")
    cols = get_tag_int(ds, "Columns")
    wc = get_tag_list(ds, "WindowCenter")
    ww = get_tag_list(ds, "WindowWidth")
    pixel_spacing = get_tag_list(ds, "PixelSpacing")
    sop_class = get_tag_str(ds, "SOPClassUID")
    num_frames = get_tag_int(ds, "NumberOfFrames") or 1
    
    if wc is not None:
        wc = wc[0] if len(wc) == 1 else wc
    if ww is not None:
        ww = ww[0] if len(ww) == 1 else ww

    is_generic_video = (
        sop_class == "1.2.840.10008.5.1.4.1.1.7" or
        sop_class == "1.2.840.10008.5.1.4.1.1.7.4"
    )
    if is_generic_video and num_frames > 1:
        sop_class = "1.2.840.10008.5.1.4.1.1.12.1"

    orientation = get_tag_list(ds, "ImageOrientationPatient")
    position = get_tag_list(ds, "ImagePositionPatient")
    
    image_type_str = get_tag_str(ds, "ImageType")
    image_type = image_type_str.split("\\") if image_type_str else None

    meta = {
        "Columns": cols,
        "Rows": rows,
        "NumberOfFrames": num_frames,
        "InstanceNumber": get_tag_int(ds, "InstanceNumber"),
        "SOPClassUID": sop_class,
        "PhotometricInterpretation": get_tag_str(ds, "PhotometricInterpretation"),
        "BitsAllocated": get_tag_int(ds, "BitsAllocated"),
        "BitsStored": get_tag_int(ds, "BitsStored"),
        "PixelRepresentation": get_tag_int(ds, "PixelRepresentation"),
        "SamplesPerPixel": get_tag_int(ds, "SamplesPerPixel") or 1,
        "HighBit": get_tag_int(ds, "HighBit"),
        "FrameOfReferenceUID": get_tag_str(ds, "FrameOfReferenceUID"),
        "Modality": "CT" if num_frames > 1 else get_tag_str(ds, "Modality"),
        "SOPInstanceUID": get_tag_str(ds, "SOPInstanceUID"),
        "SeriesInstanceUID": get_tag_str(ds, "SeriesInstanceUID"),
        "StudyInstanceUID": get_tag_str(ds, "StudyInstanceUID"),
        "SeriesDate": get_tag_str(ds, "SeriesDate"),
        "SliceThickness": get_tag_float(ds, "SliceThickness"),
    }
    if pixel_spacing:
        meta["PixelSpacing"] = pixel_spacing
    if orientation:
        meta["ImageOrientationPatient"] = orientation
    if position:
        meta["ImagePositionPatient"] = position
    if image_type:
        meta["ImageType"] = image_type
    if wc is not None:
        meta["WindowCenter"] = wc
    if ww is not None:
        meta["WindowWidth"] = ww
        
    return meta

def build_ohif_manifest(session_id: str, origin: str) -> dict:
    session_path = SESSIONS_DIR / session_id
    instances_path = session_path / "instances"
    
    if not instances_path.exists():
        return {"studies": []}
        
    rows = []
    import pydicom
    for p in instances_path.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            meta = get_instance_metadata(ds)
            meta["StudyInstanceUID"] = session_id
            rows.append(meta)
        except Exception as e:
            print(f"Error parsing metadata for {p}: {e}")
            
    if not rows:
        return {"studies": []}
        
    by_series = {}
    for r in rows:
        series_uid = r["SeriesInstanceUID"]
        if series_uid not in by_series:
            by_series[series_uid] = []
        by_series[series_uid].append(r)
        
    patient_name = ""
    patient_id = ""
    study_date = ""
    study_time = ""
    accession = ""
    age = ""
    sex = ""
    desc = ""
    
    try:
        first_file = next(instances_path.glob("*.dcm"))
        ds_full = pydicom.dcmread(first_file, stop_before_pixels=True)
        patient_name = str(ds_full.get("PatientName", ""))
        patient_id = str(ds_full.get("PatientID", ""))
        study_date = str(ds_full.get("StudyDate", ""))
        study_time = str(ds_full.get("StudyTime", ""))
        accession = str(ds_full.get("AccessionNumber", ""))
        age = str(ds_full.get("PatientAge", ""))
        sex = str(ds_full.get("PatientSex", ""))
        desc = str(ds_full.get("StudyDescription", ""))
    except Exception:
        pass
        
    modalities = list(set(r["Modality"] for r in rows if r.get("Modality")))
    
    study = {
        "StudyInstanceUID": session_id,
        "StudyDate": study_date,
        "StudyTime": study_time,
        "PatientName": patient_name,
        "PatientID": patient_id,
        "AccessionNumber": accession,
        "PatientAge": age,
        "PatientSex": sex,
        "StudyDescription": desc,
        "NumInstances": sum(r.get("NumberOfFrames", 1) or 1 for r in rows),
        "Modalities": "\\".join(modalities),
        "series": [],
    }
    
    for series_uid, ser_rows in by_series.items():
        def sort_key(r):
            pos = r.get("ImagePositionPatient")
            if pos and len(pos) >= 3:
                return pos[2]
            return r.get("InstanceNumber") or 0
            
        ser_rows.sort(key=sort_key)
        s0 = ser_rows[0]
        
        expanded_instances = []
        for r in ser_rows:
            num_frames = r.get("NumberOfFrames") or 1
            sop_instance_uid = r["SOPInstanceUID"]
            enc = urllib.parse.quote(sop_instance_uid)
            url_base = f"{origin}/api/ohif-dicom/session/{session_id}/instance/{enc}"
            
            for frame_index in range(1, num_frames + 1):
                meta_copy = dict(r)
                meta_copy["InstanceNumber"] = (r.get("InstanceNumber") or 1) + (frame_index - 1)
                
                frame_suffix = f"?frame={frame_index}" if num_frames > 1 else ""
                expanded_instances.append({
                    "metadata": meta_copy,
                    "url": f"wadouri:{url_base}{frame_suffix}",
                })
                
        study["series"].append({
            "SeriesInstanceUID": series_uid,
            "SeriesNumber": s0.get("SeriesNumber"),
            "Modality": s0["Modality"],
            "SeriesDescription": s0.get("SeriesDescription", ""),
            "SliceThickness": s0.get("SliceThickness"),
            "instances": expanded_instances,
        })
        
    return {"studies": [study]}

@app.post("/api/ohif-session")
async def create_ohif_session(
    sessionId: str = Form(...),
    origin: str = Form(...),
    files: List[UploadFile] = File(...)
):
    prune_python_sessions()
    
    session_path = SESSIONS_DIR / sessionId
    instances_path = session_path / "instances"
    instances_path.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for file in files:
        file_bytes = await file.read()
        sop_instance_uid = None
        try:
            import pydicom
            import io
            ds = pydicom.dcmread(io.BytesIO(file_bytes), stop_before_pixels=True)
            sop_instance_uid = ds.SOPInstanceUID
        except Exception:
            filename = file.filename or ""
            if filename.endswith(".dcm"):
                sop_instance_uid = filename[:-4]
            else:
                sop_instance_uid = filename
        
        if not sop_instance_uid:
            import uuid
            sop_instance_uid = f"fallback_{uuid.uuid4().hex}"
            
        safe_uid = re.sub(r'[^a-zA-Z0-9.\-_]', '_', sop_instance_uid)
        
        with open(instances_path / f"{safe_uid}.dcm", "wb") as f:
            f.write(file_bytes)
        saved_count += 1
        
    # Generate and save manifest
    try:
        manifest_json = build_ohif_manifest(sessionId, origin)
        with open(session_path / "manifest.json", "w") as f:
            json.dump(manifest_json, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate manifest: {str(e)}")
        
    with open(session_path / "meta.json", "w") as f:
        json.dump({"createdAt": int(time.time() * 1000)}, f)
        
    return {"status": "success", "sessionId": sessionId, "savedInstances": saved_count}

from fastapi import Request

@app.post("/api/ohif-session/{sid}/slice")
async def upload_session_slice(sid: str, request: Request):
    session_path = SESSIONS_DIR / sid
    instances_path = session_path / "instances"
    instances_path.mkdir(parents=True, exist_ok=True)
    
    file_bytes = await request.body()
    sop_instance_uid = None
    try:
        import pydicom
        import io
        ds = pydicom.dcmread(io.BytesIO(file_bytes), stop_before_pixels=True)
        sop_instance_uid = ds.SOPInstanceUID
    except Exception:
        filename = request.headers.get("x-filename", "slice.dcm")
        if filename.endswith(".dcm"):
            sop_instance_uid = filename[:-4]
        else:
            sop_instance_uid = filename
            
    if not sop_instance_uid:
        import uuid
        sop_instance_uid = f"fallback_{uuid.uuid4().hex}"
        
    safe_uid = re.sub(r'[^a-zA-Z0-9.\-_]', '_', sop_instance_uid)
    
    with open(instances_path / f"{safe_uid}.dcm", "wb") as f:
        f.write(file_bytes)
        
    return {"status": "success", "sopInstanceUID": sop_instance_uid}

from pydantic import BaseModel
class FinalizeRequest(BaseModel):
    origin: str

@app.post("/api/ohif-session/{sid}/finalize")
async def finalize_session(sid: str, req: FinalizeRequest):
    prune_python_sessions()
    session_path = SESSIONS_DIR / sid
    
    try:
        manifest_json = build_ohif_manifest(sid, req.origin)
        with open(session_path / "manifest.json", "w") as f:
            json.dump(manifest_json, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate manifest: {str(e)}")
        
    with open(session_path / "meta.json", "w") as f:
        json.dump({"createdAt": int(time.time() * 1000)}, f)
        
    return {"status": "success", "sessionId": sid}

@app.get("/api/ohif-session/{sid}/manifest")
async def get_ohif_manifest(sid: str):
    session_path = SESSIONS_DIR / sid
    manifest_file = session_path / "manifest.json"
    if not manifest_file.exists():
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    try:
        with open(manifest_file, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read manifest: {str(e)}")

@app.get("/api/ohif-session/{sid}/instance/{uid}")
async def get_ohif_instance(sid: str, uid: str):
    from urllib.parse import unquote
    decoded_uid = unquote(uid)
    safe_uid = re.sub(r'[^a-zA-Z0-9.\-_]', '_', decoded_uid)
    
    instance_file = SESSIONS_DIR / sid / "instances" / f"{safe_uid}.dcm"
    if not instance_file.exists():
        raise HTTPException(status_code=404, detail="Instance not found")
    
    headers = {
        "Content-Type": "application/dicom",
        "Access-Control-Allow-Origin": "*",
        "Cross-Origin-Resource-Policy": "cross-origin",
        "Cache-Control": "private, max-age=3600"
    }
    return FileResponse(path=instance_file, headers=headers, media_type="application/dicom")

# Video Analysis and Claude Vision Features
BUNNY_STREAM_API_KEY = os.getenv("BUNNY_STREAM_API_KEY", "")
BUNNY_STREAM_LIBRARY_ID = os.getenv("BUNNY_STREAM_LIBRARY_ID", "558250")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

ANALYSES_FILE = Path(__file__).parent / "video_analyses.json"

def read_analyses() -> dict:
    if ANALYSES_FILE.exists():
        try:
            with open(ANALYSES_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading analyses: {e}")
    return {}

def save_analyses(data: dict):
    try:
        with open(ANALYSES_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving analyses: {e}")

def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}

async def perform_ai_analysis(title: str, length: int) -> dict:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
    if anthropic_key:
        try:
            print(f"Querying Claude to analyze video: {title}")
            headers = {
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            prompt = (
                f"You are an expert orthopedic and neurosurgical spinal analysis AI.\n"
                f"Generate an extremely comprehensive, highly detailed, professional, and clinically accurate formal surgical dictation report for this endoscopic spinal surgery video:\n"
                f"- Video Title: {title}\n"
                f"- Duration: {length} seconds\n\n"
                f"You must return ONLY a JSON object with this exact structure (do not wrap in markdown code blocks like ```json, just return raw JSON text):\n"
                f"{{\n"
                f"  \"summary\": \"Provide a highly detailed, extensive multi-paragraph clinical narrative summary of the procedure (minimum 250-300 words). Discuss the target pathology and level (e.g. C5-C6), patient indications, specific surgical approach/access trajectory (e.g. interlaminar, transforaminal), soft tissue dissection, bone work (laminotomy/facetectomy), discectomy details, visual landmark confirmation, hemostasis, and the final clinical verification of neural element decompression.\",\n"
                f"  \"steps\": [\n"
                f"    \"Step 1: Patient positioning, sterile preparation, and localization using fluoroscopy.\",\n"
                f"    \"Step 2: Dilator placement and target portal access path verification.\",\n"
                f"    \"Step 3: ... (provide 4-6 detailed sequential steps based on the video title and standard spinal surgical techniques) ...\"\n"
                f"  ],\n"
                f"  \"findings\": [\n"
                f"    \"Anatomical landmark 1 / clinical finding (e.g., hypertrophic ligamentum flavum, herniated nucleus pulposus).\",\n"
                f"    \"Anatomical landmark 2 / surgical decompression confirmation (e.g., mobilization of nerve root, pulsation of thecal sac).\",\n"
                f"    \"Hemostasis and closure confirmation.\"\n"
                f"  ],\n"
                f"  \"timeline\": [\n"
                f"    {{\"tool\": \"endoscope\", \"start\": 0, \"end\": {length}}},\n"
                f"    {{\"tool\": \"grasper\", \"start\": 60, \"end\": 180}},\n"
                f"    ...\n"
                f"  ]\n"
                f"}}\n\n"
                f"For the 'timeline', list realistic timestamps (start and end in seconds) for various instruments (e.g., endoscope, grasper, hook, scissors, retractor, suction, radiofrequency probe, drill) used throughout the duration of {length} seconds. The endoscope should span the entire length. Return ONLY valid JSON."
            )
            
            payload = {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            async with httpx.AsyncClient(timeout=45.0) as client:
                res = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
                if res.status_code == 200:
                    res_json = res.json()
                    content = res_json.get("content", [])
                    if content and len(content) > 0:
                        text_response = content[0].get("text", "")
                        extracted = extract_json(text_response)
                        if extracted and "summary" in extracted:
                            extracted["analyzedAt"] = datetime.datetime.utcnow().isoformat() + "Z"
                            return extracted
                else:
                    print(f"Claude analysis API error: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"Failed querying Claude for video analysis: {e}")

    # Fallback to MedGemma if Claude key is missing or fails
    if MEDGEMMA_API_KEY:
        try:
            prompt = (
                f"Analyze this endoscopic spine surgery video:\n"
                f"Title: {title}\n"
                f"Duration: {length} seconds.\n\n"
                f"Provide a structured surgical report. Return ONLY a JSON object with this exact structure:\n"
                f"{{\n"
                f"  \"summary\": \"A high-level narrative summary of the procedure.\",\n"
                f"  \"steps\": [\"Step 1 description\", \"Step 2 description\", ...],\n"
                f"  \"findings\": [\"Clinical finding 1\", \"Clinical finding 2\", ...],\n"
                f"  \"timeline\": [\n"
                f"    {{\"tool\": \"endoscope\", \"start\": 0, \"end\": {length}}},\n"
                f"    {{\"tool\": \"grasper\", \"start\": 30, \"end\": 120}},\n"
                f"    ...\n"
                f"  ]\n"
                f"}}\n"
                f"List typical tools such as endoscope, grasper, hook, scissors, retractor, suction, or radiofrequency probe. Make sure times (start and end) are within the video duration. Output ONLY JSON, do not include any other text."
            )
            
            data_to_send = {
                "model": "medgemma-27b-it",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 8192,
                "temperature": 0.3
            }
            
            headers = {
                "Authorization": f"Bearer {MEDGEMMA_API_KEY}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(MEDGEMMA_API_URL, json=data_to_send, headers=headers)
                if res.status_code == 200:
                    res_json = res.json()
                    choices = res_json.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        extracted = extract_json(content)
                        if extracted and "summary" in extracted:
                            extracted["analyzedAt"] = datetime.datetime.utcnow().isoformat() + "Z"
                            return extracted
        except Exception as e:
            print(f"Error querying MedGemma for video analysis: {e}")
            
    tools_list = ["endoscope", "grasper", "hook", "scissors", "retractor", "suction"]
    timeline = []
    timeline.append({"tool": "endoscope", "start": 0, "end": length})
    
    cur = 10
    while cur < length - 20:
        tool = random.choice(tools_list[1:])
        duration = random.randint(30, min(180, length - cur - 10))
        timeline.append({"tool": tool, "start": cur, "end": cur + duration})
        cur += duration + random.randint(20, 60)
        
    mock_data = {
        "summary": f"Clinical analysis of the surgical video '{title}'. The video demonstrates a spinal procedure utilizing endoscopic techniques to achieve neural decompression.",
        "steps": [
            "Patient positioned in prone position. Endoscopic entry port established.",
            "Initial endoscopic visualization of the target spinal segment and identification of landmarks.",
            "Endoscopic decompression performed using visual guides to clear soft tissue.",
            "Visual check for decompression completion and standard port closure."
        ],
        "findings": [
            "Clear visualization of the nerve root after decompression.",
            "No active bleeding observed at the close of the procedure."
        ],
        "timeline": timeline,
        "analyzedAt": datetime.datetime.utcnow().isoformat() + "Z"
    }
    return mock_data

async def hourly_video_analysis_loop():
    await asyncio.sleep(5)
    while True:
        try:
            print("[Background Task] Running hourly video library sync...")
            api_key = os.getenv("BUNNY_STREAM_API_KEY") or BUNNY_STREAM_API_KEY
            library_id = os.getenv("BUNNY_STREAM_LIBRARY_ID") or BUNNY_STREAM_LIBRARY_ID
            
            if not api_key:
                print("[Background Task] Warning: BUNNY_STREAM_API_KEY is not configured. Skipping sync.")
            else:
                async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                    url = f"https://video.bunnycdn.com/library/{library_id}/videos?page=1&itemsPerPage=100"
                    res = await client.get(url, headers={"AccessKey": api_key, "accept": "application/json"})
                    if res.status_code == 200:
                        videos_data = res.json()
                        videos = videos_data.get("Items", [])
                        analyses = read_analyses()
                        updated = False
                        for video in videos:
                            video_id = video.get("guid") or video.get("videoId")
                            status = video.get("status")
                            title = video.get("title", "Surgical Video")
                            length = video.get("length", 0)
                            
                            if video_id and status == 4 and video_id not in analyses:
                                print(f"[Background Task] Auto-analyzing new video: {title} ({video_id})")
                                analysis_report = await perform_ai_analysis(title, length)
                                analyses[video_id] = analysis_report
                                updated = True
                        if updated:
                            save_analyses(analyses)
                    else:
                        print(f"[Background Task] Failed to fetch videos from Bunny: {res.status_code}")
        except Exception as e:
            print(f"[Background Task] Error in background loop: {e}")
        await asyncio.sleep(3600)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(hourly_video_analysis_loop())

@app.get("/api/videos/analyses")
async def get_all_analyses():
    return read_analyses()

class AnalyzeVideoRequest(BaseModel):
    videoId: str
    title: str
    length: int
    force: Optional[bool] = False

@app.post("/api/videos/analyze")
async def analyze_video(req: AnalyzeVideoRequest):
    analyses = read_analyses()
    if not req.force and req.videoId in analyses:
        return analyses[req.videoId]
    
    print(f"Triggering on-demand analysis for video (force={req.force}): {req.title} ({req.videoId})")
    report = await perform_ai_analysis(req.title, req.length)
    analyses[req.videoId] = report
    save_analyses(analyses)
    return report

class AnnotateRequest(BaseModel):
    image: str
    videoId: str

@app.post("/api/videos/annotate")
async def annotate_frame(req: AnnotateRequest):
    """Sends a video frame (base64) to Claude Vision for tool and anatomy detection, using video context."""
    api_key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
    base64_data = req.image
    if "," in base64_data:
        base64_data = base64_data.split(",")[1]
        
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="LLM Error: ANTHROPIC_API_KEY is not configured in the backend environment."
        )
        
    # Debug: Save the received image to verify what the backend is seeing
    try:
        import base64 as py_base64
        img_data = py_base64.b64decode(base64_data)
        debug_img_path = Path(__file__).parent / "debug_frame.jpg"
        with open(debug_img_path, "wb") as f:
            f.write(img_data)
        print(f"[DEBUG] Saved current frame to {debug_img_path}")
    except Exception as e:
        print(f"[DEBUG] Failed to save debug image: {e}")

    # Read clinical context from cached reports if available
    analyses = read_analyses()
    video_context = ""
    if req.videoId in analyses:
        analysis = analyses[req.videoId]
        summary = analysis.get("summary", "")
        steps = ", ".join(analysis.get("steps", []))
        findings = ", ".join(analysis.get("findings", []))
        video_context = (
            f"SURGERY PROCEDURE CONTEXT:\n"
            f"- Procedure Summary: {summary}\n"
            f"- Surgical Steps: {steps}\n"
            f"- Intraoperative Findings: {findings}\n\n"
        )
        
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 1000,
            "system": (
                "You are a specialized surgical computer vision AI. Your task is to identify both the surgical tools "
                "and the anatomical structures/landmarks in this laparoscopic/endoscopic spine surgery video frame.\n\n"
                "Endoscopic Frame Geometry Context:\n"
                "- The image is a 16:9 video frame containing a circular endoscopic view in the center.\n"
                "- The regions to the left and right of the circle, as well as the four corners, are black background margins.\n"
                "- CRITICAL: Do NOT extend bounding boxes into the black background margins. Keep the bounding boxes extremely tight, cropped closely around the actual tools and anatomical structures inside the central circular view.\n"
                "- All coordinates must be precise and lie within the active circular aperture (approx. horizontal x-coordinates 220 to 780 out of 1000).\n\n"
                "Detect the following categories:\n"
                "1. Surgical Tools (e.g., endoscope, grasper, hook, scissors, retractor, suction, probe, drill)\n"
                "2. Anatomical Structures/Landmarks (e.g., nerve root, dura/thecal sac, disc space, bone/lamina, ligamentum flavum, muscle)\n"
                "3. Interactions: If a tool is interacting with or pointing to an anatomical structure, "
                "label it using the format '{tool} pointing to {anatomy}' (e.g., 'grasper pointing to nerve root').\n\n"
                "Provide the exact label and its normalized bounding box [ymin, xmin, ymax, xmax] coordinates on a scale of 0 to 1000.\n"
                "Return ONLY a JSON object where ALL detections (both tools and anatomical structures) are returned in a single 'tools' array. Example:\n"
                "{\"tools\": [{\"label\": \"grasper pointing to nerve root\", \"box_2d\": [320, 410, 680, 520]}, {\"label\": \"nerve root\", \"box_2d\": [600, 500, 850, 700]}]}.\n"
                "Output ONLY valid JSON."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": [
                      {
                        "type": "image",
                        "source": {
                          "type": "base64",
                          "media_type": "image/jpeg",
                          "data": base64_data
                        }
                      },
                      {
                        "type": "text",
                        "text": f"{video_context}Analyze this laparoscopic/endoscopic video frame. Detect the tools, anatomical structures, and any interactions, keeping all bounding boxes tightly inside the circular view, and return them as JSON."
                      }
                    ]
                  }
            ]
        }
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            res = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            if res.status_code == 200:
                res_json = res.json()
                content = res_json.get("content", [])
                if content and len(content) > 0:
                    text_response = content[0].get("text", "")
                    print(f"[DEBUG] Claude Vision Raw Response: {text_response}")
                    extracted = extract_json(text_response)
                    if extracted:
                        # Defensive merge: combine separate lists into "tools"
                        merged_tools = []
                        if "tools" in extracted and isinstance(extracted["tools"], list):
                            merged_tools.extend(extracted["tools"])
                        
                        # Merge alternative keys if Claude separates them
                        for key in ["anatomical_structures", "structures", "anatomy", "landmarks", "anatomicalStructures", "interactions"]:
                            if key in extracted and isinstance(extracted[key], list):
                                merged_tools.extend(extracted[key])
                                
                        extracted["tools"] = merged_tools
                        print(f"[DEBUG] Merged detections output: {json.dumps(extracted)}")
                        return extracted
                raise HTTPException(
                    status_code=500,
                    detail="LLM Error: Claude Vision responded successfully, but failed to output a valid tools JSON object."
                )
            else:
                print(f"Claude API Error: {res.status_code} - {res.text}")
                err_detail = res.text
                try:
                    err_json = res.json()
                    if "error" in err_json and "message" in err_json["error"]:
                        err_detail = err_json["error"]["message"]
                except Exception:
                    pass
                raise HTTPException(
                    status_code=res.status_code,
                    detail=f"LLM Error: Claude API call failed with status {res.status_code}. Detail: {err_detail}"
                )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Failed calling Claude Vision API: {e}")
        raise HTTPException(status_code=500, detail=f"LLM Error: Failed calling Claude API. Detail: {str(e)}")



class ImagingDiagnosisRequest(BaseModel):
    caseContext: dict
    base64Image: Optional[str] = None
    provider: Optional[str] = "claude"

@app.post("/api/ai/imaging-diagnosis")
async def analyze_imaging_diagnosis(request: ImagingDiagnosisRequest):
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic API key is not configured")
    
    prompt = f"""You are a medical AI assistant specializing in spine surgery.
Analyze the following patient data:
{json.dumps(request.caseContext, indent=2)}

"""
    if request.base64Image:
        prompt += """You have been provided with an actual DICOM slice image from the viewer. 
IMPORTANT CRITICAL INSTRUCTION: Analyze the provided image meticulously. Actively look for disc herniations, foraminal narrowing, and listhesis. Extract realistic anatomical measurements (like canal diameter, disc height, and herniation size) directly from the image features you observe. Do not output 'No images provided' since you have an image."""
    else:
        prompt += """IMPORTANT CRITICAL INSTRUCTION: If there is no explicit image data or measurement data provided in the caseContext annotations, DO NOT hallucinate or guess measurements (like millimeters, degrees, or grades). Instead, explicitly state "No images provided" or "Not assessed" for any measurement or pixel analysis that cannot be directly derived from the text."""
        
    prompt += """
Generate an analysis based ONLY on the provided context (and image if provided).
Return ONLY a JSON object with this exact structure:
{
  "pixelAnalysisTags": ["tag1", "tag2"],
  "measurements": {
    "spine": {"Canal diameter": "value", "Disc height": "value"},
    "other": {"Hip": "value"}
  },
  "severityGrades": {"Pfirrmann": "value"},
  "differentials": [
    {
      "diagnosis": "Name of diagnosis",
      "probability": "High/Medium/Low",
      "reasoning": "Reasoning string"
    }
  ]
}
IMPORTANT: You MUST ensure all internal quotes inside any strings are properly escaped (e.g., using \\\") to guarantee the output is valid parsable JSON.
"""

    messages = []
    if request.base64Image:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": request.base64Image,
                    },
                },
                {"type": "text", "text": prompt}
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})

    try:
        if request.provider == "medgemma":
            if request.base64Image:
                # Step 1: Vision model extracts raw features
                vision_prompt = "Analyze this DICOM image. Extract raw Pixel Analysis Tags, Measurements, and Severity Grades. DO NOT output a diagnosis."
                vision_text = await call_medgemma_vision(prompt=vision_prompt, base64_image=request.base64Image)
                
                # Step 2: Text model synthesizes the JSON with differentials
                synthesis_prompt = f"{prompt}\n\nHere is the raw visual data extracted by the vision model:\n{vision_text}\n\nUse this visual data to construct the final JSON including the differential diagnosis."
                content = await call_medgemma_text(prompt=synthesis_prompt, system_prompt="You are an expert orthopedic AI assistant. You must output raw JSON only.")
            else:
                content = await call_medgemma_text(prompt=prompt, system_prompt="You are an expert orthopedic AI assistant. You must output raw JSON only.")
        else:
            if not anthropic_client:
                raise HTTPException(status_code=500, detail="Anthropic API key is not configured")
            response = await anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                temperature=0.2,
                system="You are an expert orthopedic and neurosurgical AI assistant. You must output raw JSON only.",
                messages=messages
            )
            content = response.content[0].text
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].strip()
        return json_repair.loads(content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ManagementRequest(BaseModel):
    caseContext: dict
    selectedDiagnosis: str
    provider: Optional[str] = "claude"

@app.post("/api/ai/management")
async def analyze_management(request: ManagementRequest):
    
    prompt = f"""You are a medical AI assistant specializing in spine surgery.
Analyze the following patient data:
{json.dumps(request.caseContext, indent=2)}
The primary diagnosis selected is: {request.selectedDiagnosis}

Generate evidence-based management options for this diagnosis.
Return ONLY a JSON object with this exact structure:
{{
  "options": [
    {{
      "title": "Name of management option (e.g. Conservative Care, Surgery)",
      "recommendationLevel": "e.g. 85% Recommendation",
      "description": "Detailed description and reasoning"
    }}
  ]
}}
CRITICAL: DO NOT use double quotes (") anywhere inside your text descriptions. If you need to quote something or emphasize text, use single quotes ('). Your entire output MUST be perfectly valid JSON without any unescaped quotes.
"""
    try:
        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            temperature=0.2,
            system="You are an expert orthopedic and neurosurgical AI assistant. You must output raw JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].strip()
        return json_repair.loads(content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class PreopPlanningRequest(BaseModel):
    caseContext: dict
    selectedDiagnosis: str
    selectedManagement: str
    provider: Optional[str] = "claude"

@app.post("/api/ai/preop-planning")
async def analyze_preop_planning(request: PreopPlanningRequest):
    
    prompt = f"""You are a medical AI assistant specializing in spine surgery.
Analyze the following patient data:
{json.dumps(request.caseContext, indent=2)}
The primary diagnosis selected is: {request.selectedDiagnosis}
The management plan selected is: {request.selectedManagement}

Generate a preop plan, surgical steps (if applicable), implant options (if applicable), and postop care.
CRITICAL: Keep your response EXTREMELY concise. Limit each array to a maximum of 3-4 short, bullet-point style items. DO NOT write long paragraphs.

Return ONLY a JSON object with this exact structure:
{{
  "preopOptimization": ["short step 1", "short step 2"],
  "surgicalSteps": ["short step 1", "short step 2"],
  "implantOptions": [
    {{"name": "Implant Name", "description": "Short details"}}
  ],
  "postopCare": ["short step 1", "short step 2"]
}}
CRITICAL: DO NOT use double quotes (") anywhere inside your text descriptions. If you need to quote something, use single quotes ('). Your entire output MUST be perfectly valid JSON.
"""
    try:
        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            temperature=0.2,
            system="You are an expert orthopedic and neurosurgical AI assistant. You must output raw JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        
        print("CLAUDE RAW OUTPUT:", content) # debug
        return json_repair.loads(content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        # return empty structure to allow frontend to not crash
        return {
            "preopOptimization": ["Error generating optimization steps"],
            "surgicalSteps": ["Error generating surgical steps"],
            "implantOptions": [],
            "postopCare": []
        }

class ChatRequest(BaseModel):
    caseContext: dict
    chatHistory: list
    userMessage: str
    provider: Optional[str] = "claude"

@app.post("/api/ai/chat")
async def analyze_chat(request: ChatRequest):
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic API key is not configured")
    
    messages = [
        {"role": "user", "content": f"You are a medical AI assistant for spine surgeons. Here is the case context:\n{json.dumps(request.caseContext, indent=2)}"}
    ]
    for msg in request.chatHistory:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    
    messages.append({"role": "user", "content": request.userMessage})

    try:
        if request.provider == "medgemma":
            prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            content = await call_medgemma_text(
                prompt=prompt,
                system_prompt="You are an expert orthopedic and neurosurgical AI assistant. Answer the surgeon's questions concisely."
            )
            return {"response": content}
        else:
            if not anthropic_client:
                raise HTTPException(status_code=500, detail="Anthropic API key is not configured")
            response = await anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                temperature=0.7,
                system="You are an expert orthopedic and neurosurgical AI assistant. Answer the surgeon's questions concisely.",
                messages=messages
            )
            return {"response": response.content[0].text}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
