import copy
import numpy as np
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.logging import get_logger
from cut_img.type1_cut import *

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
        # logger.info("dt_boxes num : {}, elapse : {}".format(
        #     len(dt_boxes), elapse))
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
            # logger.info("cls num  : {}, elapse : {}".format(
            #     len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        # logger.info("rec_res num  : {}, elapse : {}".format(
        #     len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res



def judge_type(ori_img):
    text_sys = TextSystem(utility.parse_args())
    img_shape = ori_img.shape

    top_pixel1 = int(img_shape[0] * 0.201825)
    bot_pixel1 = int(img_shape[0] * 0.474344)
    judge_img1 = ori_img[top_pixel1: bot_pixel1, :]
    # cv2.imshow('1', judge_img1)
    # cv2.waitKeyEx(0)
    dt_boxes1, rec_res1 = text_sys(judge_img1)

    for i, re in enumerate(rec_res1):
        if '确认报告' in re[0]:
            # print(dt_boxes[i])
            ratio_cf1 = dt_boxes1[i][2][1] + top_pixel1
            flag1 = 1
            return flag1, ratio_cf1

    top_pixel = int(img_shape[0] * 0.5758)
    bot_pixel = int(img_shape[0] * 0.7126567)
    judge_img = ori_img[top_pixel: bot_pixel, :]
    dt_boxes, rec_res = text_sys(judge_img)

    for i, re in enumerate(rec_res):
        if '确认报告' in re[0]:
            # print(dt_boxes[i])
            ratio_cf = dt_boxes[i][2][1] + top_pixel
            flag = 2
            return flag, ratio_cf

    return 0, 0
