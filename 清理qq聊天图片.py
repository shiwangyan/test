import re
import shutil
import os

txt_path = r'xxxxx.txt'#改成自己导出的日志txt的路径，如在同一目录下只需输入文件名
target_path = "D:\\QQtmp\\"#改成目标临时文件夹路径,复制时记得\要手动多打一个变成\\
disk = 'D'#QQ所在的硬盘，如果在D盘则改为D

if not os.path.exists(target_path+"Group\\"):
    os.mkdir(target_path+"Group\\")
if not os.path.exists(target_path+"Group2\\"):
    os.mkdir(target_path+"Group2\\")
with open(txt_path, 'r', encoding='gb2312') as f:
    for line in f:
        if "FILE_read," in line:
            matchObj = re.search(r'{0}:.+?(?=,)'.format(disk), line)
            if matchObj:
                pic_path=matchObj.group()
                print(pic_path)
                dir1 = pic_path.split('\\')[-3]
                dir2 = pic_path.split('\\')[-2]
                filename = pic_path.split('\\')[-1]
                if 'Group2' in pic_path:
                    if not os.path.exists(target_path+"Group2\\"+dir1):
                        os.mkdir(target_path+"Group2\\"+dir1)
                    if not os.path.exists(target_path+"Group2\\"+dir1+"\\"+dir2):
                        os.mkdir(target_path+"Group2\\"+dir1+"\\"+dir2)
                    try:
                        shutil.copyfile(pic_path, target_path+"Group2\\"+dir1+"\\"+dir2+"\\"+filename)
                    except:
                        continue
                else:
                    if not os.path.exists(target_path+"Group\\"+dir1):
                        os.mkdir(target_path+"Group\\"+dir1)
                    if not os.path.exists(target_path+"Group\\"+dir1+"\\"+dir2):
                        os.mkdir(target_path+"Group\\"+dir1+"\\"+dir2)
                    try:
                        shutil.copyfile(pic_path, target_path+"Group\\"+dir1+"\\"+dir2+"\\"+filename)
                    except:
                        continue