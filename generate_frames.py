import cv2
import os


'''
This function takes the video from the  time_lapse video folder and creates a database  of images
'''
#FIRST STEP IN CREATING DATABASE
def FrameCapture(hpath):

    path = "./videos/"+hpath
    vidObj = cv2.VideoCapture(path)

    count=0
    flag=1

    #let's try with extracting every 100th fram
    while flag:
        flag, image = vidObj.read()

        if flag:
            count +=1
        
        if count%20 ==0:
            filename = hpath[:-4]+"_frame_"+str(count)+".jpg"
            print("Saving file", filename)
            if not image is None:
                cv2.imwrite("./video_frames/"+filename, image)


if __name__ == "__main__":
    video_list = os.listdir('./videos')
    video_list = ['video17.mp4']
    for k in video_list:
        FrameCapture(k)