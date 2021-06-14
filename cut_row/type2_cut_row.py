import cv2

def fill_type2_PRE(xps_img):
    row_list = ['FVC', 'FEV1', 'FEV1/FVC%', 'PEF', 'FEF25-75%', 'MEF25%', 'MEF50%', 'MEF75%', 'FEV6', 'FEV1/FEV6%',
                'MIF/MEF50%', 'FEV1/VCMAX%']
    column_list = ['PRE_实测值', 'PRE_正常范围', 'PRE_Pred', 'PRE_%百分比', 'PRE_z-score', '舒张试验_实测值', '舒张试验_更改',
                   '舒张试验_%变化', '舒张试验_%百分比', '舒张试验_z-score']
    fill_excel = dict()
    add1 = xps_img.shape[0] // 11 + 1
    # 返回12行数据
    img_row_list = list()
    row_pixel = 0
    for i in range(11):
        img_row_list.append(xps_img[row_pixel: row_pixel + add1, :])
        # cv2.imshow('type2_tb', xps_img[row_pixel: row_pixel + add1, :])
        # cv2.waitKeyEx(0)
        row_pixel = row_pixel + add1
    return img_row_list

