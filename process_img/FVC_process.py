from cut_img.type1_cut import *
from cut_row.type1_cut_row import *
from result2excel.type1_result2excel import *
import tools.infer.predict_rec as predict_rec
import tools.infer.utility as utility
import os
import re


def FVC_process(ori_img, ratio_cf):
    te = predict_rec.TextRecognizer(utility.parse_args())
    ratio_rc = 0.07867731
    ratio_tb1_top = ratio_cf / ori_img.shape[0] + ratio_rc
    ratio_tb1_bot = ratio_tb1_top + 0.1471

    tb1 = cut_img_FVC(ori_img, ratio_tb1_top, ratio_tb1_bot)

    row_FVC = fill_FVC(tb1)

    FVC_list = list()
    for fvc in row_FVC:
        re_ = FVC_row_excel(fvc)
        # for r in re_:
        #     cv2.imshow('FVC', r)
        #     cv2.waitKey(0)
        tb, tr = te(re_)
        for im, f_name in zip(re_, tb):
            cv2.imwrite(os.path.join('./check_imgs', f_name[0]+'.jpg'), im)
        # print("**********")
        row_item = list()
        # print(len(tb))
        for tb_item in tb:
            # for tt in tb_item[0].split('-'):
            # for tt in tb_item[0]:
                # if tt.isalnum:
                #     continue
            tt = tb_item[0]
            if '0' in tt or '1' in tt or '2' in tt or '3' in tt or '4' in tt or '5' in tt or '6' in tt or \
                    '7' in tt or '8' in tt or '9' in tt:
                try:
                    # a = float(tt)
                    tb_item = list(tb_item)
                    tb_item[0] = float(tt)
                    tb_item = tuple(tb_item)
                except:
                    tb_item = list(tb_item)
                    # print(tt)
                    tb_item[0] = re.sub('[^0-9-.]', '', tt)
                    tb_item = tuple(tb_item)
            else:
                tb_item = list(tb_item)
                tb_item[0] = '-'
                tb_item = tuple(tb_item)

            row_item.append(tb_item[0])
            # print(tb_item)
        # print(row_item)
        FVC_list.append(row_item)
    return FVC_list, ratio_tb1_bot