import os
import glob
import httpx
import asyncio

async def test_mri_limit():
    dicom_dir = "../dicomm"
    
    # Check limits by sending from 4 up to 15 images
    for limit in range(4, 16):
        print(f"\n--- Testing with {limit} images ---")
        
        # Select 'limit' number of images
        dicom_files = [os.path.join(dicom_dir, f"IMG-0005-{str(i).zfill(5)}.dcm") for i in range(1, limit + 1)]
        
        files_to_upload = []
        for fpath in dicom_files:
            files_to_upload.append(
                ("dicom_files", (os.path.basename(fpath), open(fpath, "rb"), "application/dicom"))
            )
        
        data = {
            "input_data": "Please analyze these MRI scans and identify any abnormalities.",
        }
        
        url = "http://127.0.0.1:8000/api/orchestrate"
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(url, data=data, files=files_to_upload)
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                        # Check if the response is a canned refusal
                        if "I am sorry" in content and "cannot analyze MRI scans" in content:
                            print(f"FAILED at {limit} images: Triggered safety refusal.")
                            print(f"CONCLUSION: The maximum number of images is {limit - 1}.")
                            break
                        elif "error" in content.lower() or len(content) < 50:
                            print(f"FAILED at {limit} images: Unusual or short response.")
                            print(f"CONCLUSION: The maximum number of images is {limit - 1}.")
                            break
                        else:
                            print(f"SUCCESS at {limit} images! Provided a valid analysis.")
                            
                    except Exception as e:
                        print(f"Failed to parse JSON response: {e}")
                        break
                else:
                    print(f"API Error {response.status_code} at {limit} images.")
                    print(f"CONCLUSION: The maximum number of images is {limit - 1}.")
                    break
        finally:
            for _, file_tuple in files_to_upload:
                file_tuple[1].close()

if __name__ == "__main__":
    asyncio.run(test_mri_limit())
