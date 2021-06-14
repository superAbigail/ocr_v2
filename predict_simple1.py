import os
import re
import copy
import numpy as np
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.logging import get_logger
from cut_img.type1_cut import *
from process_img.FVC_process import FVC_process
from process_img.SVC_process import SVC_process
from process_img.info_process import info_process
from process_img.time_process import time_process
from process_img.color_process import color_process, color_process_type1_c3


# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
# sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

def sum_of_(FVC_list):
    s = 0
    for fl in FVC_list:
        for f in fl:
            if f == '-':
                s = s + 1
    return s


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



def type1_(ori_img, ratio_cf):

    # image_file_list = get_image_file_list(args.image_dir)
    # for image_file in image_file_list:

    FVC_list, ratio_tb1_bot = FVC_process(ori_img, ratio_cf)
    tb_c1, tb_c2, ratio_color2_bot = color_process(ori_img, ratio_tb1_bot)
    PRE_list, ratio_tb2_bot = SVC_process(ori_img, ratio_color2_bot)
    time_list = time_process(ori_img, ratio_cf)
    tb_c3 = color_process_type1_c3(ori_img, ratio_tb2_bot)
    tb_info = info_process(ori_img)

    if sum_of_(FVC_list) > 30:
        for i in range(len(FVC_list)):
            for ii in range(len(FVC_list[i])):
                if ii > 4:
                    FVC_list[i][ii] = '-'

    excel_tb = list()
    info_index = [2, 1, 3, 4, 5, 6, 0, 7]
    for i in info_index:
        excel_tb.append(tb_info[i][0])
    excel_tb[4] = re.sub('[^0-9.]', '', excel_tb[4])
    for i in range(10):
        for j in range(12):
            excel_tb.append(FVC_list[j][i])

    # excel_tb.append(time_list[0] + '-' + time_list[1])
    excel_tb.append(time_list)
    excel_tb.append(tb_c1[0][0])
    excel_tb.append(tb_c2[0][0])

    for i in range(5):
        for j in range(5):
            excel_tb.append(PRE_list[j][i])
    excel_tb.append(tb_c3[0][0])
    # print(excel_tb)
    return excel_tb


# if __name__ == "__main__":
#     type1_()
