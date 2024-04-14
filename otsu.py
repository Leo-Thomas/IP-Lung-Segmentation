import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

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
    img = cv2.imread(false_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)



    start_time = time.time()
    # Normalizar la imagen
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Aplicar el umbral de Otsu
    threshold_value, binary_img = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Aplicar una operación morfológica de apertura
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    result_img = img_norm - binary_img
    runtime=round(time.time() - start_time,4)


    # Calcular metricas de evaluacion
    intersection = cv2.bitwise_and(gt, result_img)
    union = cv2.bitwise_or(gt, result_img)
    iou_score = np.sum(intersection) / np.sum(union)
    dice_coef = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(result_img))

    iou.append(iou_score)
    dc.append(dice_coef)
    runtimes.append(runtime)
    plt.imshow(result_img, cmap = 'gray')
    plt.show()



print("Minimums\n")
print(f"IoUMin= {round(min(iou),4)}, DCMin= {round(min(dc),4)} y RuntimeMin= {min(runtimes)} \n")
print("Maximums\n")
print(f"IoUMax= {round(max(iou),4)}, DCMax= {round(max(dc),4)} y RuntimeMax= {max(runtimes)} \n")
print("Averages\n")
print(f"IoUAvg= {round(sum(iou)/len(iou),4)}, DCAvg= {round(sum(dc)/len(dc),4)} y RuntimeAvg= {round(sum(runtimes)/len(runtimes),4)} \n")


