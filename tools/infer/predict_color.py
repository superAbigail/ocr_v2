import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger

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

        dt_boxes = sorted_boxes(dt_boxes)

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


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


# 裁剪原始图像，变换为FVC区域
# 1_1.jpeg 1240*1754
# 261 303 345 383 425 466 507
# 1040 1072 1102
def cut_img_QC_Pre(xps_img):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.210484)
    wide2 = int(img_shape[1] * 0.408871)
    long1 = int(img_shape[0] * 0.5929304)
    long2 = int(img_shape[0] * 0.6111745)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_QC_Pre.jpg', xps_img)
    return xps_img

def cut_img_QC_Post(xps_img):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.210484)
    wide2 = int(img_shape[1] * 0.408871)
    long1 = int(img_shape[0] * 0.6111745)
    long2 = int(img_shape[0] * 0.6282782)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_QC_Post.jpg', xps_img)
    return xps_img


def cut_img_PRE_Pre(xps_img):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.210484)
    wide2 = int(img_shape[1] * 0.408871)
    long1 = int(img_shape[0] * 0.794755)
    long2 = int(img_shape[0] * 0.8124287)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_PRE_Pre.jpg', xps_img)
    return xps_img



def fill_QC(xps_img):
    add1 = xps_img.shape[0] // 6
    img_row_list = list()
    row_pixel = 0
    for i in range(6):
        img_row_list.append(xps_img[row_pixel: row_pixel + add1, :])
        row_pixel = row_pixel + add1
    return img_row_list

def QC_row_excel(xps_img):
    add1 = xps_img.shape[1] // 6
    img_row_list = list()
    row_pixel = 0
    for i in range(6):
        img_row_list.append(xps_img[:, row_pixel: row_pixel + add1])
        row_pixel = row_pixel + add1
    return img_row_list

# 接受每行的图像，然后返回其对应值，一共10个值的图像
def FVC_row_excel(row_img):
    column_list = ['PRE_实测值', 'PRE_正常范围', 'PRE_Pred', 'PRE_%百分比', 'PRE_z-score', '舒张试验_实测值', '舒张试验_更改',
                   '舒张试验_%变化', '舒张试验_%百分比', '舒张试验_z-score']
    row_ps = [0, 0.0742574, 0.227723, 0.316, 0.409241, 0.514851, 0.6113861, 0.708746, 0.8028, 0.90429, 1]
    img_shape = row_img.shape
    row_excel = list()
    for i in range(10):
        row_excel.append(row_img[:, int(img_shape[1] * row_ps[i]): int(img_shape[1] * row_ps[i+1])])
    return row_excel

# 316*107
def PRE_row_excel(row_img):
    row_ps = [0, 0.17405, 0.46202531, 0.623417724, 0.835443038, 1]
    img_shape = row_img.shape
    row_excel = list()
    for i in range(5):
        row_excel.append(row_img[:, int(img_shape[1] * row_ps[i]): int(img_shape[1] * row_ps[i + 1])])
    return row_excel


def info_row_excel(row_img):
    # r0 = [0.9041, 1]
    # r1 = [0.231, 0.3410, 0.566607, 0.6554174, 0.6971581, 0.739787, 0.79, 0.825, 0.83, 0.91, 0.952, 0.99]
    # r2 = [0.41, 0.4982]
    # row_img = row_img[0]
    row_excel = list()
    img_shape = row_img[0].shape
    r = [[0.9041, 1], [0.231, 0.3410, 0.566607, 0.6554174, 0.6971581, 0.739787, 0.79, 0.825, 0.83, 0.91, 0.952, 1],
         [0.41, 0.4982]]
    for i in range(3):
        rr_len = len(r[i])
        for ii in range(0, rr_len, 2):
            le = int(img_shape[1] * r[i][ii])
            ri = int(img_shape[1] * r[i][ii + 1])
            row_excel.append(row_img[i][:, le: ri])
    return row_excel


# 1_1.jpeg 1240*1754
# 姓名，ID，性别，年龄，体重，身高，出生日期，测试日期
def main(args):
    row_list = ['姓名', 'ID', '性别', '年龄', '体重', '身高', '出生日期', '测试日期']
    te = predict_rec.TextRecognizer(args)
    image_file_list = get_image_file_list(args.image_dir)
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img_ori = cv2.imread(image_file)
            img_QC_Pre = cut_img_PRE_Pre(img_ori)


            inf_1 = QC_row_excel(img_QC_Pre)
        if inf_1 is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        for i, p in enumerate(inf_1):
            if 0 in p:
                print(i)
            cv2.imshow('1', p)
            cv2.waitKey(0)
        tb, tr = te(inf_1)
        print(tb)
        print(tr)

if __name__ == "__main__":
    main(utility.parse_args())
