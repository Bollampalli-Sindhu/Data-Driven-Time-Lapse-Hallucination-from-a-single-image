import cv2
import os
import numpy as np
import imutils
import glob


TIMELAPSE_VIDEO_DIRECTORY = "./timelapse_videos"
IMAGE_DATABASE_DIRECTORY = "./image_database"
HISTOGRAM_BINS_TUPLE = (8, 12, 3) #subjected to fine tuning
FEATURE_CSV_FILE = "features.csv"
OUTPUT_MOVIE_NAME = None



def database_frame_generator(video_name):
    '''
    This function takes the video from the time_lapsee video folder and creates a database of images
    '''
    full_path = os.path.join(TIMELAPSE_VIDEO_DIRECTORY, video_name)
    
    video = cv2.VideoCapture(full_path)

    count = 0
    flag = 1

    #extracting every 20th frame
    
    while flag:
        
        flag, image = video.read() 
        
        if flag:
            count +=1 #increase the count of frames captured
        
        if count%20 == 0:
            frame_name = video_name[:-4]+"_frame_" + str(count) + ".jpg"
            print("Saving Frame, ", frame_name, end='\r')
            cv2.imwrite(os.path.join(IMAGE_DATABASE_DIRECTORY, frame_name), image)

def extract_feature_description(input_image):
    '''
    This function is aimed at extracting the feature vector of the given input image
    '''
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    feature_vector = []

    #compute the center of the image
    height, width = image.shape[0], image.shape[1]
    center_X, center_Y = int(width*0.5), int(height*0.5)

    #dividing the image into four segments
    segment_coordinates = [(0, center_X, 0 ,center_Y), (center_X, width, 0, center_Y), (center_X, width, center_Y, height), (0, center_X, center_Y, height)]

    #contructing an central mask for the image center
    
    axes_X, axes_Y = int(width0.75)//2, int(h*0.75)//2
    center_mask = np.zeros(image.shape[:2], dtype=np.int8)
    
    #looping over the segments and extracting the feature
    for (begin_X, end_X, begin_Y, end_Y) in segment_coordinates:
        #extract features after subtracting the center mask
        corner_mask = np.zeros(image.shape[:2], dtype=np.int8)
        corner_mask = cv2.subtract(corner_mask, center_mask)

        #extracting the histogram feature from the image and updating the feature vector
        hist = cv2.calcHist([image], [0,1,2], corner_mask, HISTOGRAM_BINS_TUPLE, [0, 180, 0, 256, 0, 256])

        #normalize the historgram
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        else:
            hist = cv2.normalize(hist, hist).flatten()

        feature_vector.extend(hist)
    
    #extract a color histogram for the center image
    hist = cv2.calcHist([image], [0,1,2], center_mask, HISTOGRAM_BINS_TUPLE, [0, 180, 0, 256, 0, 256])
    
    #normalize the historgram
    if imutils.is_cv2():
        hist = cv2.normalize(hist).flatten()
    else:
        hist = cv2.normalize(hist, hist).flatten()

    feature_vector.extend(hist)

    return feature_vector

def chi2_distance(a, b):
    eps = 1e-10
    d = 0.5 * np.sum([((a-b)**2)/(a+b+eps) for (a,b) in zip(a, b)])
    return d

def search(query_image):
    
    result_dict = {}
    current_path = "./"
    with open(os.path.join(current_path, FEATURE_CSV_FILE)) as f:
        reader = csv.reader(f)

        for row in reader:
            features = [float(x) for x in row[1:]]
            d = chi2_distance(features, query_image)
            result_dict[row[0]] = d
    f.close()

    results = sorted([(v,k) for (k, v) in result_dict.items()])

    return results[0]

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

    # scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b
  
    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc
    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    # return the color transferred image
    return transfer


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    #COPIED CODE TO DISPLAY THE PROGRESS BAR    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def create_timelapse_video(input_image, v_id):
    pos = str(v_id).find("_")
    video_id = v_id[:pos]
    video_path = "./timelapse_videos/"+video_id+".mp4"
    cap = cv2.VideoCapture(video_path)

    if cap is None:
        print ("No video found")
        exit(0)
    
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count=0
    flag, image = cap.read()
    output_video_file = "timelapse1.avi"
    print("VideoId =", video_id)
    print("VideoPath =", video_path)

    image_files = []
    printProgressBar(0, total_frame, prefix = 'Progress:', suffix = 'Complete', length = 50)
    while flag:
        flag, image = cap.read()
        print("Processing status: ", end="")
        if flag:
            count +=1
            printProgressBar(count + 1, total_frame, prefix = 'Progress:', suffix = 'Complete', length = 50)
            if count%2 == 1:
                styled_image = color_transfer(image,input_image)
                name = "styled_image"+str(count) +".jpg"
                cv2.imwrite(name, styled_image)
                read_frame = cv2.imread(name)
                image_files.append(name)
            
    clip = ImageSequenceClip(image_files, fps=30)
    clip.write_videofile(OUTPUT_MOVIE_NAME)
    for k in image_files:
        os.system("rm "+k)

if __name__ == "__main__":
    video_list = ["beach1.mp4", "city2.mp4", "sky.mp4"]

    #first create a video
    for video in video_list:
        database_frame_generator(video)

    #write the features of the image_database on the features.csv file

    file_output = open(FEATURE_CSV_FILE, "a+")

    general_image_path = os.path.join(IMAGE_DATABASE_DIRECTORY, "*.jpg")
    for images in glob.glob(general_image_path):
        #extract the image ID from the image path and load the image itself
        image_id = images[images.rfind("/")+1:]
        print("Current imageId = ", image_id, end="\r")
        im = cv2.imread(images)
        features = extract_feature_description(im)
        features = [str(f) for f in features]
        file_output.write("%s, %s\n" %(image_id, ",".join(features)))

    file_output.close()

    query_image = "./query.jpg"
    OUTPUT_MOVIE_NAME = query_image[:-4]+".mp4"
