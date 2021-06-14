from judge_type import judge_type
import openpyxl
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from predict_simple1 import *
from predict_simple2 import *
import configparser
from tqdm import tqdm
# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
# sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))


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
            # logger.info(bno, rec_res[bno])

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


def rename_xps(xps_dir='./xps', dst_dir='./imgs'):
    # xps_dir = './xps'
    # dst_dir = './imgs'
    xps_file = os.listdir(xps_dir)
    for i, file_name in enumerate(xps_file):
        os.rename(os.path.join(xps_dir, file_name), os.path.join(dst_dir, str(i)+'.jpeg'))


def xps_ocr():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    cf = configparser.ConfigParser()
    cf.read(parent_dir + './config.ini')
    secs = cf.sections()
    pa = cf.get('Address', 'dir')
    save_path = cf.get('Address', 'save_path')
    # print(pa)
    image_file_list = get_image_file_list(pa)

    for i, image_file in tqdm(enumerate(image_file_list)):
        # print(str(i+1) + ' start')
        ori_img, flag = check_and_read_gif(image_file)
        if not flag:
            ori_img = cv2.imread(image_file)
        if ori_img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue

        type_, ratio_cf = judge_type(ori_img)

        if type_ == 1:
            excel_tb = type1_(ori_img, ratio_cf)
        elif type_ == 2:
            excel_tb = type2(ori_img)
        else:
            print('error' + image_file)
        # 获取excel地址
        if type_ == 1 or type_ == 2:
            if not save_path:
                path = './x1.xlsx'
            else:
                path = save_path
            workbook = openpyxl.load_workbook(path)
            for line in [excel_tb]:
                sheet = workbook['Sheet1']
                sheet.append(line)
            workbook.save(path)
        # end_str = '100%'
        # process_bar((i+1) / len(image_file_list), start_str='', end_str=end_str, total_length=len(image_file_list))



def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m"%'   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)

# path = input('dir')
if __name__ == '__main__':
    print("正在识别，请稍后...")
    # print(os.getcwd())
    xps_ocr()
    os.system("pause")
