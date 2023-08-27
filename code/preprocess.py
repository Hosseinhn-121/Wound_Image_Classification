# imports
import os
import cv2
import pandas as pd
import numpy as np
from skimage import io
from skimage import feature
import matplotlib.pyplot as plt

class make_data:

    def __init__(self):
        print('Loading the data ...')
        self.pwat_db_template = pd.read_excel(os.getcwd() + '\\legendary-meme-main\\fonte\\risorse\\dati\\pwat_db_template.xlsx')
        print('Loading the data finished')

        print('Loading the images ...')
        self.images = io.ImageCollection(os.getcwd() + '\\legendary-meme-main\\fonte\\risorse\\immagini\\immagini\\*\\*.jpg', conserve_memory=False)
        self.segmentations = io.ImageCollection(os.getcwd() + '\\legendary-meme-main\\fonte\\risorse\\immagini\\immagini_segmentate\\*\\*.jpg', conserve_memory=False)
        print('Loading the images finished')
        self.applied_images_RGB = []
        self.applied_images_HSB = []
        self.applied_images_gray = []
        self.applied_images_RGB_around = []
        self.applied_images_HSB_around = []
        self.applied_images_gray_around = []
        self.applied_images_RGB_border_R = []
        self.applied_images_RGB_border_G = []
        self.applied_images_RGB_border_B = []
        self.applied_images_HSB_border_S = []
        self.applied_images_HSB_border_B = []
        self.applied_images_gray_border = []
        self.image_names = []
        self.reindex_list = []
        self.X = pd.DataFrame()

    def Canny_detector_HSB(self):
        print('Applying the masks for HSB border ...')
        
        for image in self.images:
            hsb = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            H,S,B = cv2.split(hsb)
            S = cv2.GaussianBlur(S, (3, 3), 0.6)
            B = cv2.GaussianBlur(B, (3, 3), 0.6)
            edges_S = feature.canny(S, sigma=3)
            edges_B = feature.canny(B, sigma=3)
            self.applied_images_HSB_border_S.append(edges_S)
            self.applied_images_HSB_border_B.append(edges_B)
            
        print('Applying the masks for HSB border finished')
    
    def Canny_detector_gray(self):
        print('Applying the masks for gray border...')
        
        for image in self.images:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = feature.canny(img, sigma=3)
            self.applied_images_gray_border.append(edges)
            
        print('Applying the masks for gray border finished')
        self.show_image(edges, "gray_border") 
    
    def Canny_detector_RGB(self):
        print('Applying the masks for RGB border ...')
        
        for image in self.images:
            (B, G, R) = cv2.split(image)
            R = cv2.GaussianBlur(R, (3, 3), 0.6)
            G = cv2.GaussianBlur(G, (3, 3), 0.6)
            B = cv2.GaussianBlur(B, (3, 3), 0.6)
            edges_R = feature.canny(R, sigma=3)
            edges_G = feature.canny(G, sigma=3)
            edges_B = feature.canny(B, sigma=3)
            self.applied_images_RGB_border_R.append(edges_R)
            self.applied_images_RGB_border_G.append(edges_G)
            self.applied_images_RGB_border_B.append(edges_B)
            
        print('Applying the masks for RGB border finished')

    def RGB_to_HSV(self):
        print('Applying the masks for HSB ...')
        
        for (image, mask) in zip(self.images, self.segmentations):
            masked_image = np.where(mask, image, mask)
            hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
            self.applied_images_HSB.append(hsv)
            
        print('Applying the masks for HSB finished')
    
    def RGB_to_HSV_around(self):
        print('Applying the masks for HSB around ...')
        
        for (image, mask) in zip(self.images, self.segmentations):
            masked_image = np.where(~mask, image, ~mask)
            hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
            self.applied_images_HSB_around.append(masked_image)
            
        print('Applying the masks for HSB around finished') 

    def RGB_to_gray(self):
        print('Applying the masks for GRAY ...')
        
        for (image, mask) in zip(self.images, self.segmentations):
            masked_image = np.where(mask, image, mask)
            gray = np.dot(masked_image[...,:3], [0.2989, 0.5870, 0.1140])
            self.applied_images_gray.append(gray)
            
        print('Applying the masks for GRAY finished')
    
    def RGB_to_gray_around(self):
        print('Applying the masks for GRAY around ...')
        
        for (image, mask) in zip(self.images, self.segmentations):
            masked_image = np.where(~mask, image, ~mask)
            gray = np.dot(masked_image[...,:3], [0.2989, 0.5870, 0.1140])
            self.applied_images_gray_around.append(gray)
            
        print('Applying the masks for GRAY around finished') 

    def RGB(self):
        print('Applying the masks for RGB ...')
        
        for (image, mask) in zip(self.images, self.segmentations):
            masked_image = np.where(mask, image, mask)
            self.applied_images_RGB.append(masked_image)
            
        print('Applying the masks for RGB finished')
        self.show_image(masked_image, "RGB_inside")  

    def RGB_around(self):
        print('Applying the masks for RGB around ...')
        
        for (image, mask) in zip(self.images, self.segmentations):
            masked_image = np.where(~mask, image, ~mask)
            self.applied_images_RGB_around.append(masked_image)
            
        print('Applying the masks for RGB around finished')
        self.show_image(masked_image, "RGB_around") 

    def Extracting_features(self, color_type, applied_images):
        # Extracting features
        print('Extracting features form the ' + color_type + ' images ...')
        X = pd.DataFrame(columns=['A'])
        Z = pd.DataFrame(columns=['B'])
        for image in applied_images:
            avg_color_per_row = np.average(image, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            std_color_per_row = np.std(image, axis=0)
            std_color = np.std(std_color_per_row, axis=0)
            X = X.append({'A': avg_color }, ignore_index=True)
            Z = Z.append({'B': std_color }, ignore_index=True)
        print('Extracting features form the ' + color_type + ' images finished')
        
        X = pd.concat([X.pop('A').apply(pd.Series)])
        Z = pd.concat([Z.pop('B').apply(pd.Series)])
        if(color_type == "RGB" or color_type == "RGB_around" or color_type == "RGB_border"):
            X = X.rename(columns={0: "R_avg_" + color_type, 1: "G_avg_" + color_type, 2: "B_avg_" + color_type})
            Z = Z.rename(columns={0: "R_std_" + color_type, 1: "G_std_" + color_type, 2: "B_std_" + color_type})
        elif(color_type == "HSB"or color_type == "HSB_around" or color_type == "HSB_border"): 
            X = X.rename(columns={0: "H_avg_" + color_type, 1: "S_avg_" + color_type, 2: "B_avg_" + color_type})
            Z = Z.rename(columns={0: "H_std_" + color_type, 1: "S_std_" + color_type, 2: "B_std_" + color_type})
        else:
            X = X.rename(columns={0: "avg_" + color_type})
            Z = Z.rename(columns={0: "std_" + color_type})
        
        X=pd.concat([X,Z],axis=1)
        
        if(color_type == "HSB" or color_type == "HSB_around"):
            X = X.drop(columns=["H_avg_"+color_type, "H_std_"+color_type])
        print(X.head())
        return X

    def show_image(self, img, title):
        plt.figure()
        if title == "gray_border":
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(title)
        plt.savefig(title + '_image.png', bbox_inches='tight')

    def prepro(self):

        self.RGB()
        self.RGB_to_HSV()
        self.RGB_to_gray()
        self.RGB_around()
        self.RGB_to_HSV_around()
        self.RGB_to_gray_around()
        self.Canny_detector_RGB()
        self.Canny_detector_HSB()
        self.Canny_detector_gray()

        # Finding filename for sorting
        self.pwat_filenames = self.pwat_db_template['Name']
        

        for image in self.images.files:
            self.image_names.append(image[image.rfind("\\") + 1 :])

        # Sorting the images based on their position in excel file
        print(self.image_names)

        # Creating dataframe
        df = pd.DataFrame({'Name': self.image_names })
        self.X = pd.concat([self.X, df],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("RGB", self.applied_images_RGB)],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("HSB", self.applied_images_HSB)],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("GRAY", self.applied_images_gray)],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("RGB_around", self.applied_images_RGB_around)],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("HSB_around", self.applied_images_HSB_around)],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("GRAY_around", self.applied_images_gray_around)],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("RGB_border", self.applied_images_RGB_border_R)],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("RGB_border", self.applied_images_RGB_border_G)],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("RGB_border", self.applied_images_RGB_border_B)],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("RGB_border", self.applied_images_HSB_border_S)],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("RGB_border", self.applied_images_HSB_border_B)],axis=1)
        self.X = pd.concat([self.X, self.Extracting_features("GRAY_border", self.applied_images_gray_border)],axis=1)
        print(self.X.head())
        merged_inner = pd.merge(left=self.X, right=self.pwat_db_template, left_on='Name', right_on='Name')
        merged_inner.to_csv(os.getcwd() + '\\legendary-meme-main\\fonte\\risorse\\dati\\future_db.csv', index = False)


aa = make_data()
aa.prepro()

