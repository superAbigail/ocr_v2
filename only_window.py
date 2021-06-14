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
import os
from multiprocessing import freeze_support

freeze_support()
import tkinter as tk
from tkinter.filedialog import askopenfilenames, askopenfilename, askdirectory
import configparser


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

        self.schedule_text = tk.StringVar()
        self.schedule_ = tk.Label(self._window, textvariable=self.schedule_text).place(x=self.button_confirm_x, y=150)

    def select_save_path(self):
        self.save_path = askopenfilename(filetypes=[('xlsx文件', '*.xlsx')])
        cf = configparser.ConfigParser()
        cf.read('./config.ini')
        cf.set("Address", "save_path", self.save_path)
        cf.write(open("./config.ini", "w"))
        self.select_save_patch_label = tk.Label(self._window, text="数据保存至：" + str(self.save_path)).place(
            x=self.label_x - 50,
            y=50)

    def select_patch(self):
        path_ = askdirectory()
        cf = configparser.ConfigParser()
        cf.read('./config.ini')
        cf.set("Address", "dir", path_)
        cf.write(open("./config.ini", "w"))
        files = os.listdir(path_)  # 读入文件夹
        num_png = len(files)  # 统计文件夹中的文件个数
        self.picture_num = num_png
        self.picture_num_label = tk.Label(self._window, text="共" + str(self.picture_num) + "张图片。").place(x=self.label_x,
                                                                                                         y=100)


window = tk.Tk()
app = XpsOcrWindow(window)
# window.w
window.mainloop()
