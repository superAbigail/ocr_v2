import cv2


def fill_QC(xps_img):
    add1 = xps_img.shape[0] // 6
    img_row_list = list()
    row_pixel = 0
    for i in range(6):
        img_row_list.append(xps_img[row_pixel: row_pixel + add1, :])
        row_pixel = row_pixel + add1
    return img_row_list


def fill_FVC(xps_img):
    row_list = ['FVC', 'FEV1', 'FEV1/FVC%', 'PEF', 'FEF25-75%', 'MEF25%', 'MEF50%', 'MEF75%', 'FEV6', 'FEV1/FEV6%',
                'MIF/MEF50%', 'FEV1/VCMAX%']
    column_list = ['PRE_实测值', 'PRE_正常范围', 'PRE_Pred', 'PRE_%百分比', 'PRE_z-score', '舒张试验_实测值', '舒张试验_更改',
                   '舒张试验_%变化', '舒张试验_%百分比', '舒张试验_z-score']
    fill_excel = dict()
    add1 = xps_img.shape[0] // 12 + 1
    # 返回12行数据
    img_row_list = list()
    row_pixel = 0
    for i in range(12):
        img_row_list.append(xps_img[row_pixel: row_pixel + add1 - 1, :])
        # cv2.imshow('type1_t1', xps_img[row_pixel: row_pixel + add1, :])
        # cv2.waitKeyEx(0)
        row_pixel = row_pixel + add1
    return img_row_list


def fill_PRE(xps_img):
    add1 = xps_img.shape[0] // 5
    img_row_list = list()
    row_pixel = 0
    for i in range(5):
        img_row_list.append(xps_img[row_pixel: row_pixel + add1, :])
        # cv2.imshow('type1_tb2', xps_img[row_pixel: row_pixel + add1, :])
        # cv2.waitKeyEx(0)
        row_pixel = row_pixel + add1
    return img_row_list


def fill_info(xps_img):
    img_row_list = list()
    row_rate = 46 / 283 / 2
    row_pixel = 0
    row_add = int(xps_img.shape[0] * row_rate)
    for i in range(4):
        if i == 1:
            row_pixel = row_pixel + row_add * 2
        else:
            img_row_list.append(xps_img[row_pixel + row_add - 3: row_pixel + row_add * 2 - 1, :])
            row_pixel = row_pixel + row_add * 2
    return img_row_list


def QC_row_excel(xps_img):
    add1 = xps_img.shape[1] // 6
    img_row_list = list()
    row_pixel = 0
    for i in range(6):
        img_row_list.append(xps_img[:, row_pixel: row_pixel + add1])
        row_pixel = row_pixel + add1
    return img_row_list