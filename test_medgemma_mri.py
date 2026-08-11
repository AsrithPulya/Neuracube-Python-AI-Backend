import os
import glob
import httpx
import asyncio

async def test_mri_analysis():
    dicom_dir = "/Users/asrith_pulya/Downloads/exammple/DICOM"
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    # Limit to 16 files max to avoid payload too large if needed, though main.py handles it
    if not dicom_files:
        print("No DICOM files found in that directory.")
        return

    print(f"Found {len(dicom_files)} MRI scans for analysis in {dicom_dir}...")
    
    files_to_upload = []
    for fpath in dicom_files:
        files_to_upload.append(
            ("dicom_files", (os.path.basename(fpath), open(fpath, "rb"), "application/dicom"))
        )
    
    data = {
        "input_data": """You have been provided with DICOM slice images from the viewer. 
IMPORTANT CRITICAL INSTRUCTION: Analyze the provided images meticulously. Actively look for disc herniations, foraminal narrowing, and listhesis. Extract realistic anatomical measurements (like canal diameter, disc height, and herniation size) directly from the image features you observe. Do not output 'No images provided' since you have an image.

Generate an analysis based ONLY on the provided images.
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
}""",
    }
    
    url = "http://127.0.0.1:8000/api/orchestrate"
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(url, data=data, files=files_to_upload)
            
            if response.status_code == 200:
                print("Analysis Successful!")
                try:
                    result = response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    print("="*60)
                    print(content if content else result)
                    print("="*60)
                except Exception as e:
                    print(f"Failed to parse JSON response: {e}")
                    print(response.text)
            else:
                print(f"Error {response.status_code}: {response.text}")
    finally:
        for _, file_tuple in files_to_upload:
            file_tuple[1].close()

if __name__ == "__main__":
    asyncio.run(test_mri_analysis())
