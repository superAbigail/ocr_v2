import cv2

from cut_img.type1_cut import cut_img_info
from cut_row.type1_cut_row import fill_info
from result2excel.type1_result2excel import info_row_excel
import tools.infer.predict_rec as predict_rec
import tools.infer.utility as utility

def info_process(ori_img):
    te = predict_rec.TextRecognizer(utility.parse_args())
    img_INFO = cut_img_info(ori_img)
    img_INFO = fill_info(img_INFO)
    inf_1 = info_row_excel(img_INFO)
    # for info1 in inf_1:
    #     cv2.imshow('1', info1)
    #     cv2.waitKeyEx(0)
    tb_info, _ = te(inf_1)

    return tb_info