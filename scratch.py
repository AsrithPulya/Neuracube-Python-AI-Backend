import json

endpoints = """
class ImagingDiagnosisRequest(BaseModel):
    caseContext: dict

@app.post("/api/ai/imaging-diagnosis")
async def analyze_imaging_diagnosis(request: ImagingDiagnosisRequest):
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic API key is not configured")
    
    prompt = f\"\"\"You are a medical AI assistant specializing in spine surgery.
Analyze the following patient data:
{json.dumps(request.caseContext, indent=2)}

Generate a pixel-level analysis simulation, measurements, severity grades, and differential diagnoses based on this context.
Return ONLY a JSON object with this exact structure:
{{
  "pixelAnalysisTags": ["tag1", "tag2"],
  "measurements": {{
    "spine": {{"Canal diameter": "value", "Disc height": "value"}},
    "other": {{"Hip": "value"}}
  }},
  "severityGrades": {{"Pfirrmann": "value"}},
  "differentials": [
    {{
      "diagnosis": "Name of diagnosis",
      "probability": "High/Medium/Low",
      "reasoning": "Reasoning string"
    }}
  ]
}}
\"\"\"
    try:
        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            temperature=0.2,
            system="You are an expert orthopedic and neurosurgical AI assistant. You must output raw JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].strip()
        return json.loads(content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ManagementRequest(BaseModel):
    caseContext: dict
    selectedDiagnosis: str

@app.post("/api/ai/management")
async def analyze_management(request: ManagementRequest):
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic API key is not configured")
    
    prompt = f\"\"\"You are a medical AI assistant specializing in spine surgery.
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
\"\"\"
    try:
        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            temperature=0.2,
            system="You are an expert orthopedic and neurosurgical AI assistant. You must output raw JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].strip()
        return json.loads(content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class PreopPlanningRequest(BaseModel):
    caseContext: dict
    selectedDiagnosis: str
    selectedManagement: str

@app.post("/api/ai/preop-planning")
async def analyze_preop_planning(request: PreopPlanningRequest):
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic API key is not configured")
    
    prompt = f\"\"\"You are a medical AI assistant specializing in spine surgery.
Analyze the following patient data:
{json.dumps(request.caseContext, indent=2)}
The primary diagnosis selected is: {request.selectedDiagnosis}
The management plan selected is: {request.selectedManagement}

Generate a preop plan, surgical steps (if applicable), implant options (if applicable), and postop care.
Return ONLY a JSON object with this exact structure:
{{
  "preopOptimization": ["step 1", "step 2"],
  "surgicalSteps": ["step 1", "step 2"],
  "implantOptions": [
    {{"name": "Implant Name", "description": "Details"}}
  ],
  "postopCare": ["step 1", "step 2"]
}}
\"\"\"
    try:
        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            temperature=0.2,
            system="You are an expert orthopedic and neurosurgical AI assistant. You must output raw JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].strip()
        return json.loads(content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    caseContext: dict
    chatHistory: list
    userMessage: str

@app.post("/api/ai/chat")
async def analyze_chat(request: ChatRequest):
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic API key is not configured")
    
    messages = [
        {"role": "user", "content": f"You are a medical AI assistant for spine surgeons. Here is the case context:\\n{json.dumps(request.caseContext, indent=2)}"}
    ]
    for msg in request.chatHistory:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    
    messages.append({"role": "user", "content": request.userMessage})

    try:
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
"""
with open('/Users/asrith_pulya/Neuracube dev/Dicom Viewer/Python_backend_middleware/endpoints.py', 'w') as f:
    f.write(endpoints)
