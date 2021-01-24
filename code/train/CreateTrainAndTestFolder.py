import os
import shutil
source = "source_path"
dest = "dest_path"
TRAIN_PATH = r"D:\PLacement\CGG\U-Net\data2\seep_detection\train_images_256"
MASK_PATH = r"D:\PLacement\CGG\U-Net\data2\seep_detection\train_masks_256"
TRAINOUT_PATH = r"D:\PLacement\CGG\U-Net\data2\train\train_images_256"
MASKOUT_PATH = r"D:\PLacement\CGG\U-Net\data2\train\train_masks_256"
TRAINOUTTEST_PATH = r"D:\PLacement\CGG\U-Net\data2\test\train_images_256"
MASKOUTTEST_PATH = r"D:\PLacement\CGG\U-Net\data2\test\train_masks_256"
files = os.listdir(TRAIN_PATH)
n = 0
os.makedirs(TRAINOUT_PATH, exist_ok=True)
os.makedirs(MASKOUT_PATH, exist_ok=True)
os.makedirs(TRAINOUTTEST_PATH, exist_ok=True)
os.makedirs(MASKOUTTEST_PATH, exist_ok=True)
for f in files:
    if os.path.splitext(f)[1] in (".tif") and  n <=690:
        shutil.copy(TRAIN_PATH +"\\"+ f, TRAINOUT_PATH)
        shutil.copy(MASK_PATH  +"\\"+f, MASKOUT_PATH)
    else:
        shutil.move(TRAIN_PATH +"\\"+f, TRAINOUTTEST_PATH)
        shutil.move(MASK_PATH +"\\"+ f, MASKOUTTEST_PATH)
    n = n + 1