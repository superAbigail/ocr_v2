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


import tkinter as tk
from tkinter.filedialog import askopenfilenames


class XpsOcrWindow:
    def __init__(self, _window):
        self.sw = _window.winfo_screenwidth()  # 屏幕宽度
        self.sh = _window.winfo_screenheight()  # 屏幕高度
        self.width = 300  # 窗口宽
        self.height = 100  # 窗口高
        self.x = str(int((self.sw - self.width) / 2))
        self.y = str(int((self.sh - self.height) / 2))
        _window.title("XPS_OCR_识别")
        _window.geometry("{}x{}+{}+{}".format(self.width, self.height, self.x, self.y))  # "窗口宽x窗口高+窗口位于屏幕x轴+窗口位于屏幕y轴"
        self.label_width = 100
        self.label_x = int((self.width - self.label_width) / 2)
        self.b = tk.Label(_window, text="XPS_OCR_识别").place(x=self.label_x, y=0)
        self.button_width = 80
        self.button_height = 50
        self.button_x = int((self.width - self.button_width) / 2)
        self.d = tk.Button(_window, text="选择图片", command=(lambda: self.select_patch())).place(x=self.button_x, y=50)

    @staticmethod
    def select_patch():
        path_ = askopenfilenames()
        str_path_list = []
        path_ = list(path_)
        for i in range(path_.__len__()):
            path = str(path_[i])
            str_path_list.append(path.replace("/", "\\"))
        # xps_ocr(utility.parse_args())
        print(str_path_list)


window = tk.Tk()
app = XpsOcrWindow(window)
window.mainloop()
