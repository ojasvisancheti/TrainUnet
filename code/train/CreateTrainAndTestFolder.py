import os
import shutil
source = "source_path"
dest = "dest_path"
TRAIN_PATH = r"Path"
MASK_PATH = r"Path"
TRAINOUT_PATH = r"Path"
MASKOUT_PATH = r"Path"
TRAINOUTTEST_PATH = r"Path"
MASKOUTTEST_PATH = r"Path"
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
