import cv2


def cut_img_FVC(xps_img, top_condi, bot_condi):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.175)
    wide2 = int(img_shape[1] * 0.66371)
    long1 = int(img_shape[0] * top_condi)
    long2 = int(img_shape[0] * bot_condi)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    # cv2.imshow('1', xps_img)
    # cv2.waitKeyEx(0)
    # cv2.imwrite('D:\\pythonProject\\xps_ocr\\imgs\\cut_FVC_v2.jpg', xps_img)
    return xps_img


def cut_img_PRE(xps_img, top_condi, bot_condi):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.1483871)
    wide2 = int(img_shape[1] * 0.40322581)
    long1 = int(img_shape[0] * top_condi)
    long2 = int(img_shape[0] * bot_condi)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    # cv2.imshow('1', xps_img)
    # cv2.waitKeyEx(0)
    # cv2.imwrite('D:\\pythonProject\\xps_ocr\\imgs\\cut_img_PRE.jpg', xps_img)
    return xps_img


def cut_img_QC_Pre(xps_img, top_condi, bot_condi):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.210484)
    wide2 = int(img_shape[1] * 0.408871)
    long1 = int(img_shape[0] * top_condi)
    long2 = int(img_shape[0] * bot_condi)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    # cv2.imshow('1', xps_img)
    # cv2.waitKeyEx(0)
    # cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_QC_Pre.jpg', xps_img)
    return xps_img


def cut_img_QC_Post(xps_img, top_condi, bot_condi):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.210484)
    wide2 = int(img_shape[1] * 0.408871)
    long1 = int(img_shape[0] * top_condi)
    long2 = int(img_shape[0] * bot_condi)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    # cv2.imshow('1', xps_img)
    # cv2.waitKeyEx(0)
    # cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_QC_Post.jpg', xps_img)
    return xps_img


# def cut_img_PRE_Pre(xps_img, top_condi, bot_condi):
#     img_shape = xps_img.shape
#     wide1 = int(img_shape[1] * 0.210484)
#     wide2 = int(img_shape[1] * 0.408871)
#     long1 = int(img_shape[0] * 0.794755)
#     long2 = int(img_shape[0] * 0.8124287)
#     # return xps_img[751: 1007, 184: 823]
#     xps_img = xps_img[long1: long2, wide1: wide2]
#     cv2.imshow('1', xps_img)
#     cv2.waitKeyEx(0)
#     # cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_PRE_Pre.jpg', xps_img)
#     return xps_img


def cut_img_info(xps_img):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.048)
    wide2 = int(img_shape[1] * 0.9556452)
    long1 = int(img_shape[0] * 0.035348)
    long2 = int(img_shape[0] * 0.1967)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    # cv2.imshow('1', xps_img)
    # cv2.waitKeyEx(0)
    # cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_info.jpg', xps_img)
    return xps_img


def cut_img_Time(xps_img, top_condi, bot_condi):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.76613)
    wide2 = int(img_shape[1] * 0.951)
    long1 = int(img_shape[0] * top_condi)
    long2 = int(img_shape[0] * bot_condi)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    # cv2.imshow('1', xps_img)
    # cv2.waitKeyEx(0)
    # cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_Time.jpg', xps_img)
    return xps_img


