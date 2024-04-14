import numpy as np
import cv2

def merge_regions(image, thresh):
    labels = np.zeros_like(image)
    label_num = 1
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] != 0:
                if y == 0 and x == 0:
                    labels[y, x] = label_num
                    label_num += 1
                elif y == 0:
                    if abs(int(labels[y, x-1]) - int(image[y, x])) > thresh:
                        labels[y, x] = label_num
                        label_num += 1
                    else:
                        labels[y, x] = labels[y, x-1]
                elif x == 0:
                    if abs(int(labels[y-1, x]) - int(image[y, x])) > thresh:
                        labels[y, x] = label_num
                        label_num += 1
                    else:
                        labels[y, x] = labels[y-1, x]
                else:
                    if abs(int(labels[y-1, x]) - int(image[y, x])) > thresh and abs(int(labels[y, x-1]) - int(image[y, x])) > thresh:
                        labels[y, x] = label_num
                        label_num += 1
                    elif abs(int(labels[y-1, x]) - int(image[y, x])) > thresh:
                        labels[y, x] = labels[y, x-1]
                    elif abs(int(labels[y, x-1]) - int(image[y, x])) > thresh:
                        labels[y, x] = labels[y-1, x]
                    else:
                        labels[y, x] = labels[y-1, x]
                        
    new_labels = np.zeros_like(labels)
    for i in range(1, label_num):
        if np.sum(labels == i) >= 100:
            new_labels[labels == i] = 255
        else:
            new_labels[labels == i] = 0
    
    return new_labels

# Cargar imagen y ground truth
img = cv2.imread('false.tif', 0)
gt = cv2.imread('true.tif', 0)

# Segmentar imagen usando region merging
seg = merge_regions(img, 10)

def DC(mask_gt, mask_pred):
  frame_normed = 255 * (mask_pred - mask_pred.min()) / (mask_pred.max() - mask_pred.min())
  mask_pred = np.array(frame_normed, np.intc)
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum 


# Calcular metricas de evaluacion
intersection = np.logical_and(gt, seg)
union = np.logical_or(gt, seg)
iou_score = np.sum(intersection) / np.sum(union)
dice_coef = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(seg))
dice_coef=DC(gt, seg)
# Mostrar resultados
cv2.imshow('Original', img)
cv2.imshow('Ground truth', gt)
cv2.imshow('Segmentation', seg)
print('Intersection over Union:', iou_score)
print('Dice Coefficient:', dice_coef)
cv2.waitKey(0)
cv2.destroyAllWindows()
