import cv2


def info_row_excel(row_img):
    # r0 = [0.9041, 1]
    # r1 = [0.231, 0.3410, 0.566607, 0.6554174, 0.6971581, 0.739787, 0.79, 0.825, 0.83, 0.91, 0.952, 0.99]
    # r2 = [0.41, 0.4982]
    # row_img = row_img[0]
    row_excel = list()
    img_shape = row_img[0].shape
    r = [[0.9041, 1], [0.263, 0.3410, 0.566607, 0.6554174, 0.6971581, 0.739787, 0.79, 0.825, 0.83, 0.91, 0.952, 1],
         [0.41, 0.4982]]
    for i in range(3):
        rr_len = len(r[i])
        for ii in range(0, rr_len, 2):
            le = int(img_shape[1] * r[i][ii])
            ri = int(img_shape[1] * r[i][ii + 1])
            row_excel.append(row_img[i][:, le: ri])
    return row_excel


# 接受每行的图像，然后返回其对应值，一共10个值的图像
def FVC_row_excel(row_img):
    column_list = ['PRE_实测值', 'PRE_正常范围', 'PRE_Pred', 'PRE_%百分比', 'PRE_z-score', '舒张试验_实测值', '舒张试验_更改',
                   '舒张试验_%变化', '舒张试验_%百分比', '舒张试验_z-score']
    row_ps = [0, 0.0742574, 0.227723, 0.316, 0.409241, 0.514851, 0.6113861, 0.708746, 0.8028, 0.90429, 1]
    img_shape = row_img.shape
    row_excel = list()
    for i in range(10):
        row_excel.append(row_img[:, int(img_shape[1] * row_ps[i]): int(img_shape[1] * row_ps[i + 1])])
    return row_excel


# 316*107
def PRE_row_excel(row_img):
    row_ps = [0, 0.17405, 0.46202531, 0.623417724, 0.835443038, 1]
    img_shape = row_img.shape
    row_excel = list()
    for i in range(5):
        row_excel.append(row_img[:, int(img_shape[1] * row_ps[i]): int(img_shape[1] * row_ps[i + 1])])
    return row_excel