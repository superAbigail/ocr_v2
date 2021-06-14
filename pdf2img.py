import os
import fitz

pdf_dir = []
pa = './'


def get_file(path_):
    docunames = os.listdir(path_)
    for docuname in docunames:
        if os.path.splitext(docuname)[1] == '.pdf':  # 目录下包含.pdf的文件
            pdf_dir.append(docuname)


def conver_img():
    for pdf in pdf_dir:
        doc = fitz.open(os.path.join(pa, pdf))
        pdf_name = os.path.splitext(pdf)[0]
        # for pg in range(doc.pageCount):
        page = doc[0]
        rotate = int(0)
        # 每个尺寸的缩放系数为2，这将为我们生成分辨率提高四倍的图像。
        zoom_x = 2.0822
        zoom_y = 2.0822
        trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pm = page.getPixmap(matrix=trans, alpha=False)
        pm.writePNG('%s.png' % pdf_name)


if __name__ == '__main__':
    get_file(pa)
    conver_img()