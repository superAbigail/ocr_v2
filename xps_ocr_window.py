# -*- coding: utf-8 -*-
"""
Created with：PyCharm
@Author： Jcsim
@Date： 2021-4-26 16:51
@Project： XPS_OCR_GUI
@File： xps_ocr_window.py
@Description：
@Python：3.8
"""
from multiprocessing import freeze_support
freeze_support()
import threading
import tkinter as tk
from tkinter import ttk
from time import sleep
from tkinter.filedialog import askopenfilenames, askopenfilename, askdirectory
from test_xps import xps_ocr
from tqdm import tqdm
import paddle
from judge_type import judge_type
import openpyxl
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from predict_simple1 import *
from predict_simple2 import *
import configparser

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


class XpsOcrWindow:
    def __init__(self, _window):
        self.str_path_list = []
        self.sw = _window.winfo_screenwidth()  # 屏幕宽度
        self.sh = _window.winfo_screenheight()  # 屏幕高度
        self.width = 600  # 窗口宽
        self.height = 300  # 窗口高
        self.x = str(int((self.sw - self.width) / 2))
        self.y = str(int((self.sh - self.height) / 2))
        self._window = _window
        self._window.title("XPS_OCR_识别")
        self._window.geometry(
            "{}x{}+{}+{}".format(self.width, self.height, self.x, self.y))  # "窗口宽x窗口高+窗口位于屏幕x轴+窗口位于屏幕y轴"
        self.label_width = 100
        self.label_x = int((self.width - self.label_width) / 2)
        self.b = tk.Label(self._window, text="XPS_OCR_识别").place(x=self.label_x, y=0)
        
        self.button_width = 80
        self.button_height = 50
        self.button_select_x = int((self.width - self.button_width) / 2)
        
        # 选择保存路径
        self.save_path = None
        self.button_select_save_path = tk.Button(self._window, text="选择数据保存地址",
                                                 command=(lambda: self.select_save_path())).place(
            x=self.button_select_x - 200, y=50
        )
        self.select_save_patch_label = None

        # 选择图片
        self.d = tk.Button(self._window, text="选择图片所在文件夹", command=(lambda: self.select_patch())).place(
            x=self.button_select_x - 200, y=100)
        self.picture_num = 0
        self.picture_num_label = None

        self.button_confirm_x = int((self.width + self.button_width) / 2)
        self.confirm = tk.Button(self._window, text="识      别", command=(lambda: self.orc())).place(
            x=self.button_confirm_x - 280, y=150)

        # self.frame = tk.Frame(self._window).place(x=self.button_confirm_x - 200, y=100)  # 使用时将框架根据情况选择新的位置
        # self.canvas = tk.Canvas(self.frame, width=120, height=30, bg="white")
        # self.canvas.place(x=self.button_confirm_x - 200, y=100)
        # self.schedule_text = tk.StringVar()
        # # 进度条以及完成程度
        # self.out_rec = self.canvas.create_rectangle(5, 5, 105, 25, outline="blue", width=1)
        # self.fill_rec = self.canvas.create_rectangle(5, 5, 5, 25, outline="", width=0, fill="red")
        # self.schedule_ = tk.Label(self.frame, textvariable=self.schedule_text).place(x=self.button_confirm_x - 200, y=100)
        # 进度条以及完成程度
        self.p1 = ttk.Progressbar(self._window, length=200, cursor='spider', mode="determinate",
                                  orient=tk.HORIZONTAL)
        self.p1.place(x=self.button_confirm_x - 200, y=150)
        self.schedule_text = tk.StringVar()
        self.schedule_ = tk.Label(self._window, textvariable=self.schedule_text).place(x=self.button_confirm_x, y=150)

    def select_save_path(self):
        self.save_path = askopenfilename(filetypes=[('xlsx文件', '*.xlsx')])
        # cf = configparser.ConfigParser()
        # cf.read('./config.ini')
        # cf.set("Address", "save_path", self.save_path)
        # cf.write(open("./config.ini", "w"))
        self.select_save_patch_label = tk.Label(self._window, text="数据保存至：" + str(self.save_path)).place(x=self.label_x - 50,
                                                                                                         y=50)

    def select_patch(self):
        print('success')
        # path_ = askopenfilenames()
        path_ = askdirectory()
        cf = configparser.ConfigParser()
        cf.read('./config.ini')
        cf.set("Address", "dir", path_)
        cf.write(open("./config.ini", "w"))
        # str_path_list = []
        files = os.listdir(path_)  # 读入文件夹
        num_png = len(files)  # 统计文件夹中的文件个数
        # path_ = list(path_)
        self.picture_num = num_png
        # for i in range(path_.__len__()):
        #     path = str(path_[i])
        #     self.str_path_list.append(path.replace("/", "\\"))
        self.picture_num_label = tk.Label(self._window, text="共" + str(self.picture_num) + "张图片。").place(x=self.label_x,
                                                                                                         y=100)
        # if self.picture_num > 0:
        #     print('start')
        #     t = threading.Thread(target=self.xps_ocr, kwargs={"path_": self.str_path_list})
        #     # 守护 !!!
        #     t.setDaemon(True)
        #     # 启动
        #     t.start()
        #     print('end')

    def orc(self):
        list_length = self.str_path_list.__len__()
        if self.picture_num > 0:
            print('start')
            t = threading.Thread(target=self.xps_ocr, kwargs={"path_": self.str_path_list})
            # 守护 !!!
            t.setDaemon(True)
            # 启动
            t.start()
            print('end')

    # 更新进度条函数
    def change_schedule(self, now_schedule, all_schedule):
        self.canvas.coords(self.fill_rec, (5, 5, 6 + (now_schedule / all_schedule) * 100, 25))
        self._window.update()
        self.schedule_text.set(str(round(now_schedule / all_schedule * 100, 2)) + '%')
        if round(now_schedule / all_schedule * 100, 2) == 100.00:
            self.schedule_text.set("完成")

    def xps_ocr(self, path_):
        cf = configparser.ConfigParser()
        cf.read('./config.ini')
        pa = cf.get('Address', 'dir')
        image_file_list = get_image_file_list(pa)
        # image_file_list = path_
        path_length = self.picture_num
        for i, image_file in enumerate(image_file_list):
            if i == 0:
                self.p1["value"] = 1
                self._window.update()
                self.schedule_text.set(str(1) + '%')
            print(str(i + 1) + ' start')
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
                if not self.save_path:
                    path = './x1.xlsx'
                else:
                    path = self.save_path
                workbook = openpyxl.load_workbook(path)
                for line in [excel_tb]:
                    sheet = workbook['Sheet1']
                    sheet.append(line)
                workbook.save(path)

            # self.change_schedule(i+1, path_length)
            self.p1["value"] = (i + 1) / path_length * 100
            self._window.update()
            self.schedule_text.set(str(round((i + 1) / path_length * 100, 2)) + '%')
            if round((i + 1) / path_length * 100, 2) == 100.00:
                self.schedule_text.set("完成")


window = tk.Tk()
app = XpsOcrWindow(window)
# window.w
window.mainloop()
