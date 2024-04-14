import cv2  
import matplotlib.pylab as plt
import numpy as np
import time

seeds=[]
def histo(im):
    m, n = im.shape
    histogram = np.zeros(256)
    for i in range(m):
        for j in range(n):
            histogram[im[i, j]] += 1
    return histogram/(m*n)


def cumulative(histogram):
    sum = np.zeros(256)
    sum[0] = histogram[0]
    for i in range(1, len(histogram)):
        sum[i] = sum[i-1] + histogram[i]    
    return sum


def histoequalization(im):
    histogram = histo(im)
    cumulativesum = cumulative(histogram)
    g = np.uint8(255 * cumulativesum)  
    Y = im.copy()
    m, n = Y.shape
    for i in range(0, m):
        for j in range(0, n):
            Y[i, j] = g[im[i, j]] 
    return Y


def Median(img):
    shape = img.shape
    img2 = np.zeros(shape)
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            pixels = [img[i-1, j-1], img[i-1, j], img[i-1, j+1], img[i, j-1],
                    img[i, j], img[i, j+1], img[i+1, j-1], img[i+1, j], img[i+1, j+1]]
            pixels.sort()
            mediana = pixels[4]
            img2[i, j] = mediana
    return img2


def DrawCoords(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        seeds.append((y, x))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (0, 0, 0), 2)
        cv2.imshow('image', img)
    

def GetCoords(img):
  cv2.imshow('image', img)
  cv2.setMouseCallback('image', DrawCoords)
  cv2.waitKey(0)
  print("Your seeds (y,x) are =", seeds)


def cont(img):
  ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY) 
  thresh = thresh.astype(np.uint8)
  contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) 
  ima = img.copy() 
  cv2.drawContours(image=ima, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
  return ima


def DC(mask_gt, mask_pred):
  frame_normed = 255 * (mask_pred - mask_pred.min()) / (mask_pred.max() - mask_pred.min())
  mask_pred = np.array(frame_normed, np.intc)
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum 

def IoU(target, prediction):
  intersection = np.logical_and(target, prediction)
  union = np.logical_or(target, prediction)
  return np.sum(intersection) / np.sum(union)

def preprocessing(img):
  img=histoequalization(img)
  img=Median(img)
  img=cont(img)
  return img


def outenhancement(img):
  kernel = np.ones((3,3),np.uint8)
  closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 4)
  dilating= cv2.dilate(img, kernel, iterations = 3)
  return closing


def RG(img, tolerance, seeds):
  movements = [(-1, -1),(0, -1),(1, -1),(1, 0),(1, 1),(0, 1),(-1, 1),(-1, 0)]
  segmented=np.zeros(img.shape)
  height, weight = img.shape
  seedList = []
  for seed in seeds:
    seedList.append(seed)
  while(len(seedList)>0):
    current=seedList.pop(0)
    segmented[current[0],current[1]]=10 
    for i in range(8):
      point=movements[i]
      xcord=current[0]+point[0]
      ycord=current[1]+point[1]
      if xcord >= 0 and ycord >= 0 and xcord < height and ycord < weight:
        Diff=abs(int(img[current[0],current[1]]) - int(img[xcord,ycord]))
        if Diff <= tolerance and segmented[xcord, ycord] == 0:
          segmented[xcord, ycord] = 10
          seedList.append((xcord, ycord))
  return segmented

def pipeline(img):
  backup = img.copy()
  tolerance=10 #This parameter should be changed according to the problem; for the CT dataset, 10 is ok.
  GetCoords(img)
  start_time = time.time()
  result=preprocessing(backup)
  result=RG(result, tolerance, seeds)
  result=outenhancement(result)
  print("Runtime =",round(time.time() - start_time,4), "seconds")
  return result

def plot(original, segmented, true):
  fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
  ax1.imshow(original, cmap="gray")
  ax2.imshow(original, cmap="gray")
  for i in seeds:
    ax2.plot(i[1],i[0], 'r+', ms=20)
  ax3.imshow(true, cmap="gray")
  ax4.imshow(segmented, cmap="gray")
  ax1.title.set_text('Original')
  ax2.title.set_text('Seeded')
  ax3.title.set_text('Ground-truth')
  ax4.title.set_text('Prediction')
  plt.show()


#img=cv2.imread("test30.tif",0) #Write the path of the image to segment
#true=cv2.imread("true30.tif",0) #Write the path of the mask

img = cv2.imread('false.tif', 0)
true = cv2.imread('true.tif', 0)

original = img.copy()
segmented=pipeline(img)
print("DC=",round(DC(true, segmented),4))
print("IoU=",round(IoU(true, segmented),4))
plot(original, segmented, true)





