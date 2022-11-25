#multistage
import cv2
import numpy as np
import os

ensemble_part1_root = input("please input ensemble part1 vis mask file root= ")
ensemble_part2_root = input("please input ensemble part2 vis mask file root= ")
ensemble_part3_root = input("please input ensemble part3 vis mask file root= ")
#ensemble_part4_root = input("please input ensemble part4 mask file root= ")
vis_output_root = input('please enter vis output root= ')
binary_output_root = input('please enter binary output root= ')

ensemble_part1_filelist = os.listdir(ensemble_part1_root)
ensemble_part1_filelist.sort(key=lambda x:int(x[:-4]))
ensemble_part2_filelist = os.listdir(ensemble_part2_root)
ensemble_part2_filelist.sort(key=lambda x:int(x[:-4]))
ensemble_part3_filelist = os.listdir(ensemble_part3_root)
ensemble_part3_filelist.sort(key=lambda x:int(x[:-4]))

zip_all_ensemble_parts = zip(ensemble_part1_filelist, ensemble_part2_filelist, ensemble_part3_filelist)
for i in zip_all_ensemble_parts:
    print(i[0])
    part1_file = ensemble_part1_root + "/" + i[0]
    part2_file = ensemble_part2_root + "/" + i[1]
    part3_file = ensemble_part3_root + "/" + i[2]

    #print(part1_file)
    #print(part2_file)
    #print(part3_file)

    img1 = cv2.imread(part1_file)
    img1 = img1.astype('float32')
    img2 = cv2.imread(part2_file)
    img2 = img2.astype('float32')
    img3 = cv2.imread(part3_file)
    img3 = img3.astype('float32')

    img4 = img1 + img2 + img3
    #print(img4)
    img4[img4<510] = 0
    img4[img4>=510] = 255
    img4 = img4.astype('uint8')
    vis_output_file = vis_output_root + '/' + i[0]
    cv2.imwrite(vis_output_file, img4)
    img4[img4>0] = 1
    binary_output_file = binary_output_root + '/' + i[0]
    cv2.imwrite(binary_output_file, img4)