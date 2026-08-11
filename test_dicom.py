import base64
import io
import pydicom
from PIL import Image
import numpy as np

def test():
    with open('../0002.DCM', 'rb') as f:
        file_bytes = f.read()
    
    dicom_io = io.BytesIO(file_bytes)
    ds = pydicom.dcmread(dicom_io)
    
    pixel_array = ds.pixel_array
    image_2d = pixel_array.astype(float)
    if image_2d.max() > 0:
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
    else:
        image_2d_scaled = image_2d
    image_2d_scaled = np.uint8(image_2d_scaled)
    
    image = Image.fromarray(image_2d_scaled)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    
    print("Success! Size:", len(buffered.getvalue()))

test()
