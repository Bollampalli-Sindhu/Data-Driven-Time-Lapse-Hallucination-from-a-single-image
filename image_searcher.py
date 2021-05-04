import numpy as np
import csv
import argparse
import cv2
import imutils
import os
import moviepy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from get_csv import *


def image_stats(image):
  (l,a,b) = cv2.split(image)
  (lMean, lStd) = (l.mean(), l.std())
  (aMean, aStd) = (a.mean(), a.std())
  (bMean, bStd) = (b.mean(), b.std())

  return lMean, lStd, aMean, aStd, bMean, bStd

def color_transfer(source, input_image):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)

    target = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB).astype("float32")
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
    
    (l, a, b) = cv2.split(target)
    
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b
  
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    return transfer

def load_frames(video):
    vidObj = cv2.VideoCapture(video)
    
    count=0
    flag=1
    images = []
    #let's try with extracting every 100th fram
    while flag:
        flag, image = vidObj.read()
        images.append(image)
        if flag:
            count += 1
            
    return images


def create_timelapse_video(input_image, image_id,path):
     
    images = load_frames(path+image_id+".mp4")
    total_frame = len(images)

    image_files = []
    count=0
    
    printProgressBar(0, total_frame, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for image in images:
        count +=1
        printProgressBar(count + 1, total_frame, prefix = 'Progress:', suffix = 'Complete', length = 50)
        if not image is None:
            styled_image = color_transfer(image,input_image)
            name = "styled_image"+str(count) +".jpg"
            cv2.imwrite(name, styled_image)
            read_frame = cv2.imread(name)
            image_files.append(name)
            
    clip = ImageSequenceClip(image_files, fps=10)
    clip.write_videofile('output.mp4')
    cv2.destroyAllWindows()
    for k in image_files:
        os.system("rm "+k)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
    ap.add_argument("-r", "--result-path", required = True, help = "Path to the result path")
    args = vars(ap.parse_args())

    query = cv2.imread(args["query"])
    query1 = cv2.resize(query,(150,150))
    
    results = get_target_image(query1,'./video_frames')

    create_timelapse_video(query, results,args["result_path"])
    
