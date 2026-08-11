import pydicom
ds = pydicom.dcmread('../0002.DCM')
print("Shape:", ds.pixel_array.shape)
