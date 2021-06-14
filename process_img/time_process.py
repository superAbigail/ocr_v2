import tools.infer.predict_rec as predict_rec
import tools.infer.utility as utility
from cut_img.type1_cut import *
import re


def time_process(ori_img, ratio_cf):
    te = predict_rec.TextRecognizer(utility.parse_args())
    ratio_time_top = ratio_cf / ori_img.shape[0] + 0.013113
    ratio_time_bot = ratio_time_top + 0.0142531

    img_TIME = cut_img_Time(ori_img, ratio_time_top, ratio_time_bot)
    tb, tr = te([img_TIME])
    ori_str = tb[0][0]
    # print(ori_str)
    FVC_time = re.sub('[^0-9-:]', '', ori_str)
    # time_list = list()
    # str_list = ori_str[1:].split('-')
    # for s in str_list:
    #     time_list.append(s.split('(')[0].split('（')[0])

    return FVC_time

def time_process_type2(ori_img):
    te = predict_rec.TextRecognizer(utility.parse_args())
    ratio_time_top = 0.20125427
    ratio_time_bot = 0.217217788

    img_TIME = cut_img_Time(ori_img, ratio_time_top, ratio_time_bot)
    tb, tr = te([img_TIME])
    ori_str = tb[0][0]
    # print(ori_str)
    # re.sub('[^0-9-:]')
    # time_list = list()
    # str_list = ori_str[1:].split('-')
    # for s in str_list:
    #     time_list.append(s.split('(')[0].split('（')[0])

    FVC_time = re.sub('[^0-9-:]', '', ori_str)
    return FVC_time