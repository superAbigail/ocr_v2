from cut_img.type1_cut import *
from cut_row.type1_cut_row import *
from result2excel.type1_result2excel import *
import tools.infer.predict_rec as predict_rec
import tools.infer.utility as utility
import os
import re


def SVC_process(ori_img, ratio_tb1_bot):
    te = predict_rec.TextRecognizer(utility.parse_args())
    # ratio_tb2_top = ratio_tb1_bot + 0.12543
    ratio_tb2_top = ratio_tb1_bot + 0.073588
    ratio_tb2_bot = ratio_tb2_top + 0.05986317

    tb2 = cut_img_PRE(ori_img, ratio_tb2_top, ratio_tb2_bot)

    row_PRE = fill_PRE(tb2)

    PRE_list = list()
    for pre in row_PRE:
        re_ = PRE_row_excel(pre)
        # for r in re:
        #     cv2.imwrite()
        tb, tr = te(re_)
        # print("**********")
        for im, f_name in zip(re_, tb):
            cv2.imwrite(os.path.join('./check_imgs', f_name[0]+'.jpg'), im)
        row_item = list()
        for tb_item in tb:
            # for tt in tb_item[0].split('-'):
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
                    tb_item[0] = re.sub('[^0-9-.]', '', tt)
                    tb_item = tuple(tb_item)
            else:
                tb_item = list(tb_item)
                tb_item[0] = '-'
                tb_item = tuple(tb_item)

            row_item.append(tb_item[0])
        # print(row_item)
        PRE_list.append(row_item)
    return PRE_list, ratio_tb2_bot