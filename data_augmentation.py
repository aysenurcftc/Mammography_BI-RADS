import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2 as cv
from matplotlib import pyplot as plt

class BasicImageAugmentation:
    def __init__(self):
        self.path = "data/train/0/00000056.png"
    

    def load_image(self):
        img = cv.imread(self.path)                             
        return img
    
    def show_image(self, image, picture_name):
        plt.imshow(image)
        plt.title(picture_name)
        plt.show()
        
    def plot_images(self, img_list, titles, rows, cols, wspace=0.5, hspace=0.5, save=False, save_path=None):
   
        fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))

        for i, ax in enumerate(axs.flat):
            ax.imshow(img_list[i])
            ax.set_title(titles[i])
            ax.axis('off')
        
        if save:
            if save_path:
                plt.savefig(save_path)
                print(f"Plot kaydedildi: {save_path}")
            else:
                plt.savefig("plot.png")
                print("Plot kaydedildi: plot.png")

        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.show()
        
    
    def horizontal_flipping(self, img):
        Horizontal_Flipping_Transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1) 
        ])
        Flipping_Img = Horizontal_Flipping_Transformation(img)
        #self.show_image(img, 'Original Image')
        #self.show_image(Flipping_Img, 'Flipped Image')
        return Flipping_Img
     
       
    def vertically_flipping(self, img):
        Vertical_Flipping_Transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(p=1) 
        ])
        # Testing The Transformation...
        Flipping_Img = Vertical_Flipping_Transformation(img)
        #self.show_image(img, 'Original Image')
        #self.show_image(Flipping_Img, 'Flipped Image')
        return Flipping_Img
      
       
    def rotate_transformation(self, img, degrees):
        Rotate_Transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=degrees)
        ])

        Rotated_Img = Rotate_Transformation(img)
        #self.show_image(img, 'Original Image')
        #self.show_image(Rotated_Img, 'Rotated Image') 
        return Rotated_Img
            
    
    def color_transformation(self, img):
        Color_Transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=(0.1,0.6), contrast=1,saturation=0, hue=0.4)
        ])

        # Testing The Transformation...
        Transformed_Img = Color_Transformation(img)
        #self.show_image(img, 'Original Image')
        #self.show_image(Transformed_Img, 'Transformed Image')
        return Transformed_Img
   
        
    

img = BasicImageAugmentation()
m_img = img.load_image()
img1 = img.horizontal_flipping(m_img)
img2= img.rotate_transformation(m_img, 30)
img3= img.rotate_transformation(m_img, 120)
img4 = img.vertically_flipping(m_img)
img5 = img.color_transformation(m_img)



img_list = [img1, img2, img3, img4, img5, m_img]
titles = ["Horizontal Flipping", "Rotated by 30", "Rotated by 120", "Vertical Flipping", "Color Transformation", "Original Image"]
img.plot_images(img_list, titles, rows=2, cols=3, wspace=0.5, hspace=0.5, save=True, save_path="outputs/")









