from cut_img.type2_cut import *
from cut_row.type2_cut_row import *
from result2excel.type1_result2excel import *
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import os
import re


def type2_table_process(ori_img):
    te = predict_rec.TextRecognizer(utility.parse_args())
    tb_3 = cut_img_type2_PRE(ori_img)
    row_type2_PRE = fill_type2_PRE(tb_3)
    type2_PRE_list = list()
    for t2_pre in row_type2_PRE:
        re_ = FVC_row_excel(t2_pre)
        tb, tr = te(re_)
        # print("**********")
        for im, f_name in zip(re_, tb):
            cv2.imwrite(os.path.join('./check_imgs_t2', f_name[0]+'.jpg'), im)
        row_item = list()
        # print(len(tb))
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
            # print(tb_item)
        # print(row_item)
        type2_PRE_list.append(row_item)
    return type2_PRE_list