import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

class OCR_Model:
    def __init__(self, img):
        self.image = img.copy()
        self.image2 = img.copy()
    
    def ocr_text(self):
            text = pytesseract.image_to_string(self.image) 
            return text
    
    def get_grayscale(self):
        self.image =  cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def get_grayscale2(self):
        self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(self.image2, lower_red, upper_red)
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(self.image2, lower_red, upper_red)
        mask = mask0+mask1
        output_img = self.image2.copy()
        output_img[np.where(mask == 0)] = 0
        output_hsv = self.image2.copy()
        output_hsv[np.where(mask == 0)] = 0
        self.image2 = output_hsv
        self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

    def remove_nosie(self):
        blured1 = cv2.medianBlur(self.image, 3)
        blured2 = cv2.medianBlur(self.image, 51)
        divided = np.ma.divide(blured1, blured2).data
        normed = np.uint8(255*divided/divided.max())
        self.image = normed
    def thresheding(self):
        _, threshed = cv2.threshold(self.image, 100, 255, cv2.THRESH_OTSU)
        self.image = threshed
    def showImage(self):
        plt.imshow(self.image, cmap='gray')
        plt.show()
    def morpgh_image(self):
        self.get_grayscale2()
        self.image = self.image2
        self.remove_nosie()
        self.thresheding()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))
        self.image = cv2.morphologyEx(
            self.image, cv2.MORPH_OPEN, kernel, iterations=1)
        self.image = cv2.morphologyEx(
            self.image, cv2.MORPH_CLOSE, kernel, iterations=2)
    def get_text(self):
        self.get_grayscale()
        self.remove_nosie()
        self.thresheding()
        out = []
        out.append(self.ocr_text().strip().lower().split(' ')[0])
        out.append(self.get_text2().strip().lower().split(' ')[0])
        return out
    def get_text2(self):
        self.morpgh_image()
        return self.ocr_text()


img = cv2.imread('img1.jpg')
model = OCR_Model(img)
print(model.get_text())
