##＃#＃用搜尋找val=>test or test＝>val

import numpy as np
import cv2
import os
import pandas as pd
import csv

#cal val_dataset mean IoU

#read all GT mask
GT_dir_path = "./datasets/0630_split_dataset/0630_test_binarymask/"
#GT_mask_list = os.listdir(GT_dir_path)
#GT_mask_list.sort()
#print(GT_mask_list) #case_027_left_Post-12W_M201901151656590490002_0481.png之後順序跟pred_mask順序不同! 以下從annotation裡讀回來順序和檔名
#從annotation讀回順序和檔名後取代GT_mask_list,因為pred的順序也是依照annotation的順序
import json
with open('./datasets/0630_split_dataset/annotations/0630_test_annotations.json', newline='') as jsonfile:
    data = json.load(jsonfile)
    filenamelist = []
    for i in range(len(data["images"])):
        #print(data["images"][i])
        #print(data["images"][i]['id'])
        name = data["images"][i]['file_name'].split("/")[1]
        #print(name)
        filenamelist.append(name)
    #print('filenamelist=',filenamelist)

#read all pred mask

##################################################################2選1
pred_mask_dir_path = "./SOLO_output_predict_binary_mask/"
#1007ensemble 用下面的路徑,一般用上面的路徑.
#pred_mask_dir_path = "./ensemble-multi-stages-multi-models-binary-test/"
##################################################################2選1

pred_mask_list = os.listdir(pred_mask_dir_path)
print(len(pred_mask_list))

##################################################################2選1
##pred_mask_list.sort()
pred_mask_list.sort(key=lambda x:int(x[:-4])) #solo output的檔案會有排序的問題,因此需要做處理！ #這邊算出來的IoU與SOLO_cal_average_iou_score.py的算出來有差一點點！ 不知道為啥！紀錄日期：20210421 -> 0714已解決排序問題 -> cv2方法算出來比solo本身cocoeval算出來的高一些些些!0.00幾
#print(pred_mask_list)
##################################################################2選1


zip_lists = zip(filenamelist, pred_mask_list)
print(len(filenamelist))
print(len(pred_mask_list))
#print(list(zip_lists)[0])

iou_list = []
pixelacc_list = []
pair_list = []
for pair in list(zip_lists):
    print(pair)
    pair_list.append(pair)
    #print(pair[0])
    gt = cv2.imread(GT_dir_path + pair[0])
    m = cv2.imread(pred_mask_dir_path + pair[1])
    intersection = np.logical_and(gt, m)
    union = np.logical_or(gt, m)
    iou_score = np.sum(intersection) / np.sum(union)
    iou_list.append(iou_score)
    #print(iou_list)
    pixelacc = np.sum(intersection) / np.sum(gt.astype(bool))
    #print(pixelacc)
    pixelacc_list.append(pixelacc)

#print(len(iou_list))
#print(iou_list)

listtest = []
num = 0
for i in iou_list:
    #print(num,i)
    listtest.append((num,i))
    num += 1
print('iou list=',listtest)
average_iou_score = sum(iou_list) / len(iou_list)
print("Average IoU Score: ", average_iou_score)
#print(pair_list)
#average_pixelacc = sum(pixelacc_list) / len(pixelacc_list)
#print("Average pixel acc. : ", average_pixelacc)
df = pd.DataFrame(list(zip(pair_list, listtest)), columns = ['filename_to_id', 'id_to_iou'])
print(df)
df.to_csv("filename_to_id_to_iou.csv", index=False)