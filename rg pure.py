import cv2  
import matplotlib.pylab as plt
import numpy as np
import time
import os

seeds=[]

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

def pipeline(img):
  backup = img.copy()
  tolerance=10 #This parameter should be changed according to the problem; for the CT dataset, 10 is ok.
  GetCoords(img)
  start_time = time.time()
  result=RG(backup, tolerance, seeds)
  runtime=round(time.time() - start_time,4)
  return result, runtime

def plot(original, segmented, true):
  fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
  ax1.imshow(original, cmap="gray")
  ax2.imshow(original, cmap="gray")
  for i in seeds:
    ax2.plot(i[1],i[0], 'r+', ms=20)
  ax3.imshow(true, cmap="gray")
  ax4.imshow(segmented, cmap="gray")
  #ax1.title.set_text('Original')
  #ax2.title.set_text('Seeded')
  #ax3.title.set_text('Ground-truth')
  #ax4.title.set_text('Prediction')
  plt.show()



imgs_path = "C:/Users/rleot/OneDrive/Desktop/IP LUNG/aux1"
masks_path = "C:/Users/rleot/OneDrive/Desktop/IP LUNG/aux2"
iou=[]
dc=[]
runtimes=[]


for filename in os.listdir(imgs_path):
    # Concatenamos el nombre del archivo a la ruta de la carpeta
    false_path = os.path.join(imgs_path, filename)
    true_path = os.path.join(masks_path, filename)
    # Abrimos la imagen con OpenCV
    img = cv2.imread(false_path, 0)
    gt = cv2.imread(true_path, 0)
 
    segmented, runtime=pipeline(img)

    # Calcular metricas de evaluacion
    intersection = np.logical_and(gt, segmented)
    union = np.logical_or(gt, segmented)
    iou_score = np.sum(intersection) / np.sum(union)
    dice_coef = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(segmented))
    dcc=DC(gt, segmented)
    ioouuu=IoU(gt, segmented)
    iou.append(ioouuu)
    dc.append(dcc)
    runtimes.append(runtime)
    plt.imshow(segmented, cmap = 'gray')
    plt.show()



# print("Minimums\n")
# print(f"IoUMin= {round(min(iou),4)}, DCMin= {round(min(dc),4)} y RuntimeMin= {min(runtimes)} \n")
# print("Maximums\n")
# print(f"IoUMax= {round(max(iou),4)}, DCMax= {round(max(dc),4)} y RuntimeMax= {max(runtimes)} \n")
# print("Averages\n")
# print(f"IoUAvg= {round(sum(iou)/len(iou),4)}, DCAvg= {round(sum(dc)/len(dc),4)} y RuntimeAvg= {round(sum(runtimes)/len(runtimes),4)} \n")


# #img=cv2.imread("test30.tif",0) #Write the path of the image to segment
# #true=cv2.imread("true30.tif",0) #Write the path of the mask

# img = cv2.imread('false.tif', 0)
# true = cv2.imread('true.tif', 0)

# original = img.copy()
# segmented=pipeline(img)
# # print("DC=",round(DC(true, segmented),4))
# # print("IoU=",round(IoU(true, segmented),4))
# # plot(original, segmented, true)

# # Mostrar resultados
# cv2.imshow('Original', img)
# cv2.imshow('Ground truth', true)
# cv2.imshow('Segmentation', segmented)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
