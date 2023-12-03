import cv2
import numpy as np
from matplotlib import pyplot as plt

OUTPUT_DIR = "outputs/"

def main():
    motherboard_img = cv2.imread("motherboard_image.JPEG", cv2.IMREAD_COLOR)

    motherboard_img_gray = cv2.cvtColor(motherboard_img, cv2.COLOR_BGR2GRAY)
    motherboard_img_hsv = cv2.cvtColor(motherboard_img, cv2.COLOR_RGB2HSV)

    edges = cv2.Canny(motherboard_img_gray,50,200)
    edges = cv2.dilate(edges,None, iterations = 4)

    # ret, thresh = cv2.threshold(motherboard_img_gray, 150, 300, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    contour_img = np.zeros_like(motherboard_img)

    cv2.drawContours(image=contour_img, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED)

    contour_img_hsv = cv2.cvtColor(contour_img, cv2.COLOR_RGB2HSV)
    lower_grey = np.array([0, 0, 0])
    upper_grey = np.array([255, 255, 255])
    grey_mask = cv2.inRange(contour_img_hsv, lower_grey, upper_grey)

    

    motherboard_img_np = np.float32(motherboard_img_gray)
    # dst = cv2.cornerHarris(thresh, 5, 5, 0.04)

    masked_img = cv2.bitwise_and(motherboard_img,  contour_img, mask = grey_mask)


    cv2.imwrite(OUTPUT_DIR + "motherboard_output.jpeg", masked_img)
    



if __name__ == "__main__":
    main()