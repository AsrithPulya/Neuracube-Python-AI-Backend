import os
import httpx
import json
import re

def robust_json_loads(text):
    original_text = text
    text = text.strip()
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1:
        text = text[first_brace:last_brace + 1]
    
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r',\s*([\]}])', r'\1', text)
    text = text.replace('\n', ' ').replace('\r', '')
    try:
        return json.loads(text)
    except Exception as e:
        print("--- RAW TEXT FROM MODEL ---")
        print(original_text)
        print("---------------------------")
        raise e

def test_clinician_override_and_annotations():
    dicom_dir = "/Users/asrith_pulya/Downloads/dicomm"
    
    # 1. Select representative slices from the three series
    files_to_upload = [
        os.path.join(dicom_dir, "IMG-0001-00001.dcm"),
        os.path.join(dicom_dir, "IMG-0004-00011.dcm"),
        os.path.join(dicom_dir, "IMG-0005-00001.dcm")
    ]
    
    print("=" * 85)
    print("🏥 CLINICAL OVERRIDE & DICOM ANNOTATIONS AI INTEGRATION TEST")
    print("=" * 85)
    print("ATTENDING NEUROSURGEON: Dr. Alexander Vance")
    print("SCENARIO: Surgeon corrects MedGemma outputs and adds viewer measurements.")
    print("-" * 85)
    
    print("STEP 1: SIMULATING SURGEON CORRECTION DESK (MULTI-LEVEL)...")
    # Simulate MedGemma Multi-level results which the doctor corrected
    # L4-L5: corrected from F2 to F3, Lewandrowski from Type II to Type III (Foraminal)
    levelInputs = {
        "L4-L5": {
            "fapdis": {
                "facetAngle": "F3", # Doctor-corrected to F3 (46-65 degrees)
                "anteriorPathology": ["A0"],
                "posteriorPathology": ["P2", "P3"], # Doctor added P3 (foraminal)
                "dorsalMigration": "D0",
                "inferiorMigration": "I0",
                "superiorMigration": "S0"
            },
            "lewandrowski": {
                "stenosisType": "III", # Doctor-corrected to Type III (Foraminal Stenosis)
                "multiLevel": True,
                "recurrentStenosis": False
            }
        },
        "L5-S1": {
            "fapdis": {
                "facetAngle": "F2",
                "anteriorPathology": ["A1"],
                "posteriorPathology": ["P1"],
                "dorsalMigration": "D0",
                "inferiorMigration": "I0",
                "superiorMigration": "S0"
            },
            "lewandrowski": {
                "stenosisType": "I",
                "multiLevel": True,
                "recurrentStenosis": False
            }
        }
    }
    print(" -> L4-L5: Surgeon corrected FAPDIS Facet Angle to F3 (Transverse)")
    print(" -> L4-L5: Surgeon corrected Lewandrowski Stenosis to Type III (Foraminal)")
    print("-" * 85)
    
    print("STEP 2: SIMULATING CORNERSTONE VIEWER ANNOTATIONS...")
    # Simulate two measurements made by the doctor in the active viewport
    simulated_annotations = [
        {
            "toolName": "Angle",
            "label": "Left L4-L5 Facet Angulation",
            "data": {
                "angle": 55.4
            }
        },
        {
            "toolName": "Length",
            "label": "L4-L5 foraminal canal height",
            "data": {
                "length": 4.5
            }
        }
    ]
    
    # Serialize simulated annotations (mimicking upgraded buildCaseContext() logic)
    annotations_context = "\n[SURGEON VIEWER ANNOTATIONS]\n"
    for idx, ann in enumerate(simulated_annotations):
        label = ann["label"]
        tool = ann["toolName"]
        val = ""
        if tool == "Angle":
            val = f"Angle: {ann['data']['angle']}°"
        elif tool == "Length":
            val = f"Length: {ann['data']['length']}mm"
        annotations_context += f"- Annotation #{idx+1}: Tool: \"{tool}\" (Label: \"{label}\") {val}\n"
        print(f" -> Annotation #{idx+1} added: {tool} measurement '{label}' = {val}")
    print("-" * 85)
    
    print("STEP 3: COMPILING CONSOLIDATED SURGICAL CASE CONTEXT...")
    # Serialize levels context
    levels_context = ""
    for lvl_name in ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]:
        inputs = levelInputs.get(lvl_name)
        if inputs:
            f = inputs["fapdis"]
            l = inputs["lewandrowski"]
            levels_context += f"""
[LEVEL: {lvl_name}]
- Facet Angle: {f['facetAngle']}
- Anterior Pathology: {', '.join(f['anteriorPathology']) or 'None'}
- Posterior Pathology: {', '.join(f['posteriorPathology']) or 'None'}
- Dorsal Migration: {f['dorsalMigration']}
- Inferior Migration: {f['inferiorMigration']}
- Superior Migration: {f['superiorMigration']}
- Lewandrowski Stenosis: Type {l['stenosisType']}
"""
            
    case_context = f"""[CASE CONTEXT]
Patient: 62yo Female, BMI 27.5
Primary Level: L4-L5, Side: Right
StudyInstanceUID: SI-ML-900

[MULTILEVEL DIAGNOSTIC SUMMARY]
{levels_context}
{annotations_context}
[CLINICAL ALGORITHM OVERVIEW]
FAPDIS Recommendation: IL (Confidence 85%)
Awake Candidacy: Good
ODI: 45% (Severe Disability)
VAS Back: 7/10  VAS Leg: 8/10
[END CONTEXT]"""

    print(case_context)
    print("-" * 85)
    
    print("STEP 4: RUNNING PRE-OP AI SURGICAL STRATEGY SUMMARY WITH DOCTOR FEEDBACK...")
    print("MedGemma is planning the consolidated surgical strategy based on the surgeon's edits & annotations...")
    print("-" * 85)
    
    strategy_prompt = f"""You are an expert spinal surgeon and clinical decision support assistant.
You are planning a FULL ENDOSCOPIC spine surgery.
You are given a case summary including surgeon-corrected measurements (FAPDIS algorithm), custom viewer annotations, and a sequence of DICOM frames.
Analyze the pathology progression across the provided slice sequence in the context of the algorithm inputs and surgeon annotations.

CRITICAL INSTRUCTIONS:
1. Incorporate the surgeon's custom viewer annotations (such as facet angles or canal height measurements) directly into your anatomical assessment.
2. Verify if the surgeon's corrected values (e.g. F3 facet angle and Type III foraminal stenosis at L4-L5) match the visual evidence in the multi-frame sequence.
3. Suggest the optimal endoscopic entry approach (Transforaminal or Interlaminar). Note that a F3 facet angle and Type III foraminal stenosis strongly justify a Transforaminal (TF) decompression at L4-L5.

JSON OUTPUT FORMAT:
Reply ONLY with a JSON object matching this exact shape. Do not include markdown code fences or conversational markers.

{{
  "reasoning": "Clinical reasoning incorporating surgeon edits and annotations",
  "narrative": "Clinical summary emphasizing surgical pathology and annotations",
  "keyFindings": ["finding 1"],
  "contraindications": ["item"],
  "suggestedApproach": "Surgical justification for TF, IL, or ENDO-LIF approach",
  "confidenceLevel": "High" | "Moderate" | "Low"
}}

{case_context}"""

    opened_files = []
    files_payload = []
    
    try:
        for path in files_to_upload:
            f = open(path, "rb")
            opened_files.append(f)
            files_payload.append(
                ("dicom_files", (os.path.basename(path), f, "application/dicom"))
            )
            
        target_url = "http://127.0.0.1:8000/api/orchestrate"
        with httpx.Client(timeout=120.0) as client:
            res = client.post(
                target_url,
                data={"input_data": strategy_prompt, "algorithm_data": "AI Pre-Op Strategy Override Run"},
                files=files_payload
            )
            
        if res.status_code != 200:
            print(f"🔴 ERROR: {res.text}")
            return
            
        data = res.json()
        raw_strat = data["choices"][0]["message"]["content"]
        strategy_result = robust_json_loads(raw_strat)
        
        print("🟢 MedGemma Surgical Strategy Generated Successfully!")
        print("-" * 85)
        print("Surgical Narrative:")
        print(strategy_result.get("narrative"))
        print("\nClinical Reasoning:")
        print(strategy_result.get("reasoning"))
        print("\nKey Surgical Findings:")
        for f in strategy_result.get("keyFindings", []):
            print(f" - {f}")
        print("\nContraindications:")
        for c in strategy_result.get("contraindications", []):
            print(f" - {c}")
        print("\nSuggested Approach & Justification:")
        print(strategy_result.get("suggestedApproach"))
        print(f"Confidence Level: {strategy_result.get('confidenceLevel')}")
        
    except Exception as e:
        print(f"🔴 ERROR: Simulation failed: {e}")
    finally:
        for f in opened_files:
            f.close()
    print("=" * 85)

if __name__ == "__main__":
    test_clinician_override_and_annotations()
