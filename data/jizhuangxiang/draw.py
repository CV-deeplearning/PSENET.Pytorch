#-*-coding:utf8-*-
import os
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw

def get_image_list(image_dir, suffix=['jpg','png']):
    '''get all image path ends with suffix'''
    if not os.path.exists(image_dir):
        print("PATH:%s not exists" % image_dir)
        return []
    imglist = []
    for root, sdirs, files in os.walk(image_dir):
        if not files:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.split('.')[-1] in suffix:
                imglist.append(filepath)
    return imglist


if __name__ == "__main__":
    img_list = get_image_list("train/img") 
    for img_path in tqdm(img_list):
        img = Image.open(img_path)
        dr = ImageDraw.Draw(img)
        txt_name = "gt_" + os.path.basename(img_path).replace(".jpg", ".txt").replace(".jpg", ".txt")
        txt_path = os.path.join("train/gt", txt_name)

        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.split(",")[:-1]
            rect = [float(i) for i in parts]
            dr.polygon(rect, outline="red")
        img.save(os.path.join("temp", os.path.basename(img_path)))
