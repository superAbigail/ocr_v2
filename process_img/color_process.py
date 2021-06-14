from cut_img.type1_cut import cut_img_QC_Pre, cut_img_QC_Post
from cut_row.type1_cut_row import QC_row_excel
import tools.infer.predict_rec as predict_rec
import tools.infer.utility as utility
from cut_img.type2_cut import *
import numpy as np


def color_process(ori_img, ratio_tb1_bot):
    te = predict_rec.TextRecognizer(utility.parse_args())
    ratio_color1_top = ratio_tb1_bot + 0.0211
    ratio_color1_bot = ratio_color1_top + 0.012543
    ratio_color2_top = ratio_color1_bot + 0.0057
    ratio_color2_bot = ratio_color2_top + 0.012543

    img_C1 = cut_img_QC_Pre(ori_img, ratio_color1_top, ratio_color1_bot)
    img_C1 = QC_row_excel(img_C1)
    img_C2 = cut_img_QC_Post(ori_img, ratio_color2_top, ratio_color2_bot)
    img_C2 = QC_row_excel(img_C2)

    c1 = None
    c2 = None

    for i, p in enumerate(img_C1):
        # cv2.imshow('1', p)
        # cv2.waitKeyEx(0)
        if 0 in p:
            c1 = p[:, 2:25, :]

    for i, p in enumerate(img_C2):
        # cv2.imshow('1', p)
        # cv2.waitKeyEx(0)
        if 0 in p:
            c2 = p[:, 2:25, :]

    if isinstance(c1, np.ndarray):
        c1_gray = cv2.cvtColor(c1, cv2.COLOR_RGB2GRAY)
        # _, c1_binary = cv2.threshold(c1_gray, 150, 255, cv2.THRESH_TOZERO)
        # kernel = np.ones((1, 1), np.uint8)

        # erosion = cv2.erode(c1_binary, kernel)
        # cv2.imshow('1', c1_gray)
        # cv2.waitKeyEx(0)
        c1 = cv2.cvtColor(c1_gray, cv2.COLOR_GRAY2RGB)
        tb_c1, _ = te([c1])
        if tb_c1 == 'c':
            tb_c1 = 'C'
    else:
        tb_c1 = '-'

    # cv2.imshow('1', c1)
    # cv2.waitKeyEx(0)

    if isinstance(c2, np.ndarray):
        c2_gray = cv2.cvtColor(c2, cv2.COLOR_RGB2GRAY)
        # _, c2_binary = cv2.threshold(c2_gray, 170, 255, cv2.THRESH_TOZERO)
        c2 = cv2.cvtColor(c2_gray, cv2.COLOR_GRAY2RGB)
        tb_c2, _ = te([c2])
        # cv2.imshow('1', c2)
        # cv2.waitKeyEx(0)
        if tb_c2 == 'c':
            tb_c2 = 'C'
        ratio_ = ratio_color2_bot
    else:
        tb_c2 = '-'
        ratio_ = ratio_color1_bot

    # cv2.imshow('1', c2_gray)
    # cv2.waitKeyEx(0)

    return tb_c1, tb_c2, ratio_


def color_process_type1_c3(ori_img, ratio_tb2_bot):
    te = predict_rec.TextRecognizer(utility.parse_args())
    ratio_color3_top = ratio_tb2_bot + 0.03706
    ratio_color3_bot = ratio_color3_top + 0.012543

    img_C3 = cut_img_QC_Pre(ori_img, ratio_color3_top, ratio_color3_bot)
    img_C3 = QC_row_excel(img_C3)

    c3 = None

    for i, p in enumerate(img_C3):
        if 0 in p:
            c3 = p[:, 2:25, :]

    if isinstance(c3, np.ndarray):
        tb_c3, _ = te([c3])
    else:
        tb_c3 = '-'

    return tb_c3


def color_process_type2(ori_img):
    te = predict_rec.TextRecognizer(utility.parse_args())

    img_C1 = cut_type2_QC_Pre(ori_img)
    img_C1 = QC_row_excel(img_C1)
    img_C2 = cut_type2_QC_Post(ori_img)
    img_C2 = QC_row_excel(img_C2)

    color_list = list()
    for i, p in enumerate(img_C1):
        if 0 in p:
            color_list.append(p[:, 2:25, :])

    for i, p in enumerate(img_C2):
        if 0 in p:
            color_list.append(p[:, 2:25, :])

    tb_c1, _ = te([color_list[0]])
    if tb_c1 == 'c':
        tb_c1 = 'C'
    if len(color_list) == 2:
        tb_c2, _ = te([color_list[1]])
        if tb_c2 == 'c':
            tb_c2 = 'C'
    else:
        tb_c2 = [['-']]
    return tb_c1, tb_c2