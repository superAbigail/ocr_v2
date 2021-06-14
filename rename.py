import os
import sys
import time


def rename_xps(xps_dir='./', dst_dir='./'):
    # xps_dir = './xps'
    # dst_dir = './imgs'
    xps_file = os.listdir(xps_dir)
    for i, file_name in enumerate(xps_file):
        if file_name.endswith('.jpeg') or file_name.endswith('jpg') or file_name.endswith('png'):
            os.rename(os.path.join(xps_dir, file_name), os.path.join(dst_dir, now+str(i)+'.jpeg'))


now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
rename_xps('./imgs', './imgs')
print('更改成功')