import cv2
import numpy as np
from matplotlib import pyplot as plt

OUTPUT_DIR = "outputs/"

def main():
    motherboard_img = cv2.imread("motherboard_image.JPEG", cv2.IMREAD_COLOR)

    motherboard_img_gray = cv2.cvtColor(motherboard_img, cv2.COLOR_RGB2GRAY)

    motherboard_img_gray = cv2.GaussianBlur(motherboard_img_gray, (47, 47), 4)

    motherboard_img_gray = cv2.adaptiveThreshold(motherboard_img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

    # lower_gray = 100
    # upper_gray = 100
    # gray_mask = cv2.inRange(motherboard_img_gray, lower_gray, upper_gray)
    # mask_inv_gray = cv2.bitwise_not(gray_mask)
    
    # motherboard_img = cv2.bitwise_and(motherboard_img,  motherboard_img, mask = mask_inv_gray)

    edges = cv2.Canny(motherboard_img_gray, 1, 1)
    edges = cv2.dilate(edges,None, iterations = 7)
    
    contours, _ = cv2.findContours(image=edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        print(area)
    #     if area < 70000:
    #         cv2.drawContours(edges, [c], 0, 255, -1)
    #     continue

    contour_img = np.zeros_like(motherboard_img)

    cv2.drawContours(image=contour_img, contours=[max(contours, key = cv2.contourArea)], contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED)
    
    masked_img = cv2.bitwise_and(contour_img,  motherboard_img)


    print(len(contours), "objects were found in this image.")


    cv2.imwrite(OUTPUT_DIR + "motherboard_output.jpeg", masked_img)
    



if __name__ == "__main__":
    main()