import glob
import os.path
import cv2

if __name__ == '__main__':
    for file in glob.glob("data/capture_screen/raw/*.jpeg"):
        img = cv2.imread(file)
        height, width = img.shape[:2]
        rate = 0.7
        img = cv2.resize(img, (int(rate * width), int(rate * height)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        file_name = os.path.basename(file)
        cv2.imwrite(f"data/capture_screen/compression/{file_name}", img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
