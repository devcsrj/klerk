import layoutparser.models as lpmodel
import layoutparser.visualization as lpviz
import layoutparser.ocr as lpocr
from PIL import Image

import cv2
import numpy as np


# get grayscale image
def get_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(img):
    return cv2.medianBlur(img, 5)


# thresholding
def thresholding(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


# erosion
def erode(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(img, kernel, iterations=1)


def preprocess(img):
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.bilateralFilter(img, 5, 75, 75)
    # img = cv2.medianBlur(img, 3)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return img


# opening - erosion followed by dilation
def opening(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(img):
    return cv2.Canny(img, 100, 200)


# skew correction
def deskew(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(img, template):
    return cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)


if __name__ == '__main__':
    # image = cv2.imread('preprocessing/journal-1-p0.png')

    # gray = get_grayscale(image)
    # cv2.imwrite('journal-1-p0-get_grayscale.png', gray)

    # thresh = thresholding(gray)
    # cv2.imwrite('journal-1-p0-thresholding.png', thresh)
    #
    # dilate = dilate(gray)
    # cv2.imwrite('journal-1-p0-dilate.png', dilate)
    #
    # erode = erode(gray)
    # cv2.imwrite('journal-1-p0-erode.png', erode)
    #
    # open = opening(gray)
    # cv2.imwrite('journal-1-p0-opening.png', open)
    #
    # canny = canny(gray)
    # cv2.imwrite('journal-1-p0-canny.png', canny)
    #
    # opthresh = opening(thresholding(gray))
    # cv2.imwrite('journal-1-p0-opthresh.png', opthresh)
    #
    # ppp = preprocess(gray)
    # cv2.imwrite('journal-1-p0-ppp.png', ppp)

    image = Image.open("preprocessing/journal-1-p0.png")
    # model = lpmodel.AutoLayoutModel('lp://efficientdet/PubLayNet')
    model = lpmodel.Detectron2LayoutModel(
        config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',  # In model catalog
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},  # In model`label_map`
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]  # Optional
    )
    layout = model.detect(image)
    print(layout.to_dict())

    drawn = lpviz.draw_box(image, layout, box_width=5, show_element_id=True, show_element_type=True)
    drawn.save('journal-1-p0-layout.png')

    ocr_agent = lpocr.TesseractAgent()
    for layout_region in layout:
        image_segment = layout_region.crop_image(image)
        text = ocr_agent.detect(image_segment)
        print(text)
