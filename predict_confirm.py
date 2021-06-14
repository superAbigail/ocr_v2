import os
import sys
import openpyxl
import cv2
import copy
import numpy as np
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from cut_img.type1_cut import *
from cut_row.type1_cut_row import *
from result2excel.type1_result2excel import *

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'



logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        logger.info("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.info("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res



def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    te = predict_rec.TextRecognizer(args)
    for image_file in image_file_list:
        ori_img, flag = check_and_read_gif(image_file)
        if not flag:
            ori_img = cv2.imread(image_file)
        if ori_img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue

        dt_boxes, rec_res = text_sys(ori_img)
        for i, re in enumerate(rec_res):
            if '确认报告' in re[0]:
                # print(dt_boxes[i])
                ratio_cf = dt_boxes[i][2][1]
                break
        ratio_rc = 0.07867731
        ratio_tb1_top = ratio_cf / ori_img.shape[0] + ratio_rc
        ratio_tb1_bot = ratio_tb1_top + 0.1471

        ratio_tb2_top = ratio_tb1_bot + 0.12543
        ratio_tb2_bot = ratio_tb2_top + 0.05986317

        ratio_time_top = ratio_cf / ori_img.shape[0] + 0.013113
        ratio_time_bot = ratio_time_top + 0.0142531

        tb1 = cut_img_FVC(ori_img, ratio_tb1_top, ratio_tb1_bot)
        tb2 = cut_img_PRE(ori_img, ratio_tb2_top, ratio_tb2_bot)
        # cv2.imshow('1', tb1)
        # cv2.waitKeyEx(0)
        # cv2.imshow('2', tb2)
        # cv2.waitKeyEx(0)

        row_FVC = fill_FVC(tb1)
        row_PRE = fill_PRE(tb2)

        # info
        img_INFO = cut_img_info(ori_img)
        img_INFO = fill_info(img_INFO)
        inf_1 = info_row_excel(img_INFO)

        # time
        img_TIME = cut_img_Time(ori_img, ratio_time_top, ratio_time_bot)
        tb, tr = te([img_TIME])
        ori_str = tb[0][0]
        time_list = list()
        str_list = ori_str[1:].split('-')
        for s in str_list:
            time_list.append(s.split('(')[0].split('（')[0])
        print(time_list)

        # color 换列表
        ratio_color1_top = ratio_tb1_bot + 0.0211
        ratio_color1_bot = ratio_color1_top + 0.012543
        ratio_color2_top = ratio_color1_bot + 0.0057
        ratio_color2_bot = ratio_color2_top + 0.012543
        ratio_color3_top = ratio_tb2_bot + 0.03706
        ratio_color3_bot = ratio_color3_top + 0.012543

        img_C1 = cut_img_QC_Pre(ori_img, ratio_color1_top, ratio_color1_bot)
        img_C1 = QC_row_excel(img_C1)
        img_C2 = cut_img_QC_Post(ori_img, ratio_color2_top, ratio_color2_bot)
        img_C2 = QC_row_excel(img_C2)
        img_C3 = cut_img_PRE_Pre(ori_img, ratio_color3_top, ratio_color3_bot)
        img_C3 = QC_row_excel(img_C3)

        color_list = list()
        for i, p in enumerate(img_C1):
            if 0 in p:
                color_list.append(p)

        for i, p in enumerate(img_C2):
            if 0 in p:
                color_list.append(p)

        for i, p in enumerate(img_C3):
            if 0 in p:
                color_list.append(p)
        c1 = color_list[0]
        tb_c1, _ = te([c1])
        tb_c2, _ = te([color_list[1]])
        tb_c3, _ = te([color_list[2]])

        tb_info, _ = te(inf_1)
        print(tb_info)

        FVC_list = list()
        for fvc in row_FVC:
            re = FVC_row_excel(fvc)
            tb, tr = te(re)
            print("**********")
            row_item = list()
            # print(len(tb))
            for tb_item in tb:
                for tt in tb_item[0].split('-'):
                    if tt.isalnum:
                        continue
                    else:
                        tb_item[0] = '-'
                        break
                row_item.append(tb_item[0])
                # print(tb_item)
            print(row_item)
            FVC_list.append(row_item)

        PRE_list = list()
        for pre in row_PRE:
            re = PRE_row_excel(pre)
            tb, tr = te(re)
            print("**********")
            row_item = list()
            for tb_item in tb:
                for tt in tb_item[0].split('-'):
                    if tt.isalnum:
                        continue
                    else:
                        tb_item[0] = '-'
                        break
                row_item.append(tb_item[0])
            print(row_item)
            PRE_list.append(row_item)

        row_excel = FVC_row_excel(row_FVC[0])
        test_boxes, test_rec = te(row_excel)

        excel_tb = list()
        info_index = [2, 1, 3, 4, 5, 6, 0, 7]
        for i in info_index:
            excel_tb.append(tb_info[i][0])

        for i in range(10):
            for j in range(12):
                excel_tb.append(FVC_list[j][i])

        excel_tb.append(time_list[0] + '-' + time_list[1])
        excel_tb.append(tb_c1[0][0])
        excel_tb.append(tb_c2[0][0])

        for i in range(5):
            for j in range(5):
                excel_tb.append(PRE_list[j][i])
        excel_tb.append(tb_c3[0][0])
        print(excel_tb)
        path = 'D:\\pythonProject\\PaddleOCR-release-2.0\\x1.xlsx'
        workbook = openpyxl.load_workbook('D:\\pythonProject\\PaddleOCR-release-2.0\\x1.xlsx')
        for line in [excel_tb]:
            sheet = workbook['Sheet1']
            sheet.append(line)
        workbook.save(path)  # 保存工作簿


if __name__ == "__main__":
    main(utility.parse_args())
