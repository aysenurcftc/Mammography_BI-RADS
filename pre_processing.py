import pydicom 
import numpy as np
import os
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers import apply_windowing

class PreprocessingDicom:
    def __init__(self) -> None:
        pass
    
    
    
    def read_dicom(self, file_path, output_path):
        #Load a DICOM image using pydicom
        dicom_image = pydicom.dcmread(file_path)
        
        #headers of the file
        print(dicom_image)
        
        #get the pixel data from the image
        image = dicom_image.pixel_array
        print(f"image shape: {image.shape=}")
        print(f"{image.min()=}")
        print(f"{image.max()=}")
        
        return dicom_image, image
    
    
    
    
    def standart_normalization(self, dicom_image, image):
        if dicom_image.PhotometricInterpretation == "MONOCHROME1":
            image = np.amax(image) - image
        else:
            image = image - np.min(image)
            
        if np.max(image) != 0:
            image = image / np.max(image)
        image=(image * 255).astype(np.uint8)

        return image
    
    
    
    def plot_image(self, img):
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.show()
        
        
   
    def window_image(self, dicom_image, image):
        
        # This line is the only difference in the two functions
        data = apply_windowing(image, dicom_image)
        
        if dicom_image.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        else:
            data = data - np.min(data)
            
        if np.max(data) != 0:
            data = data / np.max(data)
        data=(data * 255).astype(np.uint8)

        
        return data
    
    
    
    def crop_image(self, image, display=False):

        # Create a mask with the background pixels
        mask = image == 0
        # Find the brain area
        coords = np.array(np.nonzero(~mask))

        top_left = np.min(coords, axis=1)

        bottom_right = np.max(coords, axis=1)

        # Remove the background
        croped_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        return croped_image
    
    
       
    
medical_image = PreprocessingDicom()
dicom_image, image = medical_image.read_dicom("data/1-1.dcm", "outputs/")

crop_image_result = medical_image.crop_image(image)
medical_image.plot_image(image)
medical_image.plot_image(crop_image_result)


img_norm = medical_image.standart_normalization(dicom_image, image)
#medical_image.plot_image(img_norm)


mam_image = medical_image.window_image(dicom_image, image)
#medical_image.plot_image(mam_image)



# Plot the images
 
"""
fig, axes = plt.subplots(nrows=1, ncols=2,sharex=False, sharey=True, figsize=(14, 10))
ax = axes.ravel()
ax[0].set_title(f'Standard normalization')
ax[0].imshow(img_norm, cmap='gray');
ax[1].set_title(f'With windowing')
ax[1].imshow(mam_image, cmap='gray');
ax[0].axis("off")
plt.show()

 """




        

