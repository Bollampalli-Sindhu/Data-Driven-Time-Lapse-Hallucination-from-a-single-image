# Data Driven Time Lapse Hallucination from a single image

This project aims at creating a Time Lapse Hallucination video of a given image using the database constructed from the various timelapse hallucination videos acquired over time. The internals of the project is basically two modules working together sequentially.

## Preprocessing the Data:
In this part we have extracted frames from the timelapse videos that we have accumulated and stored them in a separate folder.

## Module 1
In this module the image feature has been studied and image matching is carried of using the technique called the Bags of Visual Word.  

## Module 2

In this module the style transfer is taking place as explained in the paper titled "Color Transfer Between Images" by Reinhart et all. 

#### Steps to execute

* If the database is updated run

  ```
  python generate_csv.py
  python generate_frames.py

  ```
* To generate a video run

  ```
  python image_searcher.py -i index.csv -q query.jpg -r image_database
  
  ```
  
  
The project was submitted as a part for the requirement for the fullfilment of course work of Computer Vision in the spring session of IIIT Hyderabad in the year 2021. The project is the joint effort of the collective works of
* Likhitha Gaddi
* B Sindhu
* Devershi Chandra

under the guidance of Mr. Prajwal Krishna (TA) and Dr. Anoop Namboodiri (Associate Professor CVIT, IIITH)
