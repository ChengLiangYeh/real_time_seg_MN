##＃#＃用搜尋找val=>test or test＝>val

import numpy as np
import cv2
import os
from pathlib import Path

cwd_mask = Path.cwd()
cwd_mask = cwd_mask / 'ensemble-multi-model-vis-val'
#print(cwd_mask)
cwd_img = Path.cwd()
cwd_img = cwd_img / 'datasets/0630_split_dataset/0630_val/JPEGImages'
#print(cwd_img)
cwd_vis_result = Path.cwd()
save_path = str(cwd_vis_result / 'post_analysis_img_result')
os.makedirs(save_path, exist_ok=True)


####
cwd_gt_img = Path.cwd()
cwd_gt_img = cwd_gt_img / 'datasets/0630_split_dataset/0630_val_binarymask'
####


#read all GT mask
GT_dir_path = "./datasets/0630_split_dataset/0630_val_binarymask/"
#print(GT_mask_list) #case_027_left_Post-12W_M201901151656590490002_0481.png之後順序跟pred_mask順序不同! 以下從annotation裡讀回來順序和檔名
#從annotation讀回順序和檔名後取代GT_mask_list,因為pred的順序也是依照annotation的順序
import json
with open('./datasets/0630_split_dataset/annotations/0630_val_annotations.json', newline='') as jsonfile:
    data = json.load(jsonfile)
    filenamelist = []
    for i in range(len(data["images"])):
        name = data["images"][i]['file_name'].split("/")[1]
        filenamelist.append(name)
    #print('filenamelist=',filenamelist)

pred_mask_dir_path = "./ensemble-multi-model-vis-val/"
pred_mask_list = os.listdir(pred_mask_dir_path)
pred_mask_list.sort(key=lambda x:int(x[:-4]))
#print(pred_mask_list)

zip_lists = zip(filenamelist, pred_mask_list)
log = []
start = 0
for pair in list(zip_lists):
    #print(pair)
    img_root = cwd_img / pair[0]
    mask_root = cwd_mask / pair[1]
    #print(img_root)
    #print(mask_root)
    img = cv2.imread(str(img_root), cv2.IMREAD_GRAYSCALE)
    img_3d = cv2.imread(str(img_root))
    mask = cv2.imread(str(mask_root), cv2.IMREAD_GRAYSCALE)


    ####
    gt_img_root = cwd_gt_img / pair[0]
    gt_img = cv2.imread(str(gt_img_root), cv2.IMREAD_GRAYSCALE)
    ####

    '''
    #cal area
    area = sum(sum(mask))
    print(area)
    area = area * 0.000025 #換算成cm^2
    print(area)
    '''

    mask = mask * 255


    ####
    gt_img = gt_img * 255
    #### 


    #cal moment and calculate x,y coordinate of centroid
    M = cv2.moments(mask)
    #print(M)
    if M["m00"] == 0:
        cX = 0
        cY = 0
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    area = M["m00"] * 0.000025
    print(area)
    #print("Centroid X coordinate=",cX)
    #print("Centroid Y coordinate=",cY)
    #Vis: put text and highlight the center
    #cv2.circle(img, (cX, cY), 2, (255, 0, 0), -1)
    cv2.circle(img_3d, (cX, cY), 2, (0, 0, 255), -1)  #紅色:pred, 綠色:GT
    #cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    ####draw gt binary mask centroid
    M2 = cv2.moments(gt_img)
    cX2 = int(M2["m10"] / M2["m00"])
    cY2 = int(M2["m01"] / M2["m00"])
    #cv2.circle(img, (cX2, cY2), 2, (0, 255, 0), -1)
    cv2.circle(img_3d, (cX2, cY2), 2, (0, 255, 0), -1)
    #### 


    #find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    #print(contours)
    #print(contours[0])
    #print(_)
    #cv2.drawContours(img,contours[0],-1,(255,0,0),2)
    if contours == []:
        print('no predicted contour.')
    else:
        cv2.drawContours(img_3d,contours[0],-1,(0,0,255),2)
        #cv2.imshow('1', img)
        #cv2.waitKey(0)


    ####draw gt binary mask contour
    contours2, __ = cv2.findContours(gt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    #cv2.drawContours(img,contours2[0],-1,(0,255,0),2)
    cv2.drawContours(img_3d,contours2[0],-1,(0,255,0),2)
    ####

    #cal perimeter
    if contours == []:
        perimeter = 0
    else:
        perimeter = cv2.arcLength(contours[0], True)
        perimeter = perimeter * 0.005 #換算成cm
        #print("Perimeter = ", perimeter, "(pixel)")

    #cal circularity
    import math
    pi = math.pi
    #print(pi)
    if perimeter == 0:
        circularity = 0
    else:
        circularity = (4 * pi * area) / (perimeter**2)
        #print("Circularity = ", circularity, "(based on pixel)")

    #save vis-image
    cv2.imwrite(save_path + '/' +pair[0], img_3d)
    start += 1
    print('analysis img. ',start)

    #logger
    temp = []
    temp.append(pair[0])
    temp.append([cX,cY])
    temp.append(area)
    temp.append(perimeter)
    temp.append(circularity)
    log.append(temp)

import pandas as pd 
df = pd.DataFrame(log, columns =['img_name', 'centroid_coordinate', 'area', 'perimeter', 'circularity'], dtype = float)
#print(df)

csv_save_path = Path.cwd()
csv_save_path = str(csv_save_path / 'post_analysis_csv_result')
os.makedirs(csv_save_path, exist_ok=True)
df.to_csv(csv_save_path + '/analysis_csv_result.csv', index=False)
