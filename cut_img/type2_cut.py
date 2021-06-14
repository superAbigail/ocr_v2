import cv2


def cut_img_type2_PRE(xps_img):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.1871)
    wide2 = int(img_shape[1] * 0.67581)
    long1 = int(img_shape[0] * 0.7349)
    long2 = int(img_shape[0] * 0.8700114)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    # cv2.imshow('1', xps_img)
    # cv2.waitKeyEx(0)
    # cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_type2_Pre.jpg', xps_img)
    return xps_img

def cut_type2_QC_Pre(xps_img):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.210484)
    wide2 = int(img_shape[1] * 0.408871)
    long1 = int(img_shape[0] * 0.8854047)
    long2 = int(img_shape[0] * 0.8991)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    # cv2.imshow('1', xps_img)
    # cv2.waitKeyEx(0)
    # cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_QC_Pre.jpg', xps_img)
    return xps_img


def cut_type2_QC_Post(xps_img):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.210484)
    wide2 = int(img_shape[1] * 0.408871)
    long1 = int(img_shape[0] * (0.8854047 + 0.019))
    long2 = int(img_shape[0] * (0.8991 + 0.019))
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    # cv2.imshow('1', xps_img)
    # cv2.waitKeyEx(0)
    # cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_QC_Pre.jpg', xps_img)
    return xps_img