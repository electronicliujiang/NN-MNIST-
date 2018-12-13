import cv2
import numpy as np
import os

save_dir="image3/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def filenames(dir):
    L=[]
    #num=[]
    for root,dirs,files in os.walk(dir):
        for file in files:
            L.append(os.path.join(root,file))
            #num.append(int(file[3]))
    #print(L)
    #print(num)
    return L

def process(files):
    for n,file in enumerate(files):
        gray_img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(gray_img, (28,28))
        ret, thresh = cv2.threshold(new_img, 125, 255, cv2.THRESH_BINARY)
        print(thresh.shape)
        for i in range(thresh.shape[0]):
            for j in range(thresh.shape[1]):
                thresh[i][j] = int(255 - thresh[i][j])
                if thresh[i][j] > 125:
                    thresh[i][j] = 255
                else:
                    thresh[i][j] = 0
        #cv2.imshow("img",thresh)
        cv2.imwrite(save_dir+"pic%i.jpg"%n, thresh)




def main():
    process_dir = "image2/"
    files=filenames(process_dir)
    process(files)


if __name__=="__main__":
    main()
    #cv2.waitKey(0)