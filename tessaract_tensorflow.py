###Install ORCB
###Move the included orcb.traineddata file (in the data folder) to your tesseract/shared folder - this will be deep in system information

###The 4 passports provided with this repository are public access and do not include any private information that puts anyone at risk. They come from this website: https://www.consilium.europa.eu/prado/en/prado-documents/AFG/A/docs-per-category.html

###Most Functional, minimal version
###Note that this requires tesseract to be installed externally, and the ORCB library.

#pip install pytesseract pyttsx3 typing blend_modes pyspellchecker pdf2image pillow passporteye matplotlib openpyxl imutils numpy scipy pandas multiprocess opencv-python mtcnn

#cd "~/GitHub/image-super-resolution/"
#pip install -e .

# importing neccessary libraries

import math
import cv2
import numpy as np
import math
import pyttsx3
import pytesseract
import re
import time
from typing import Tuple, Union
from pytesseract import Output
from blend_modes import divide
from spellchecker import SpellChecker
import requests
from urllib.request import urlopen
import sys
import pandas as pd
import pdf2image
from pdf2image import convert_from_path, convert_from_bytes
from io import BytesIO
from PIL import Image
from scipy.ndimage import interpolation as inter
from passporteye import read_mrz
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
#detector = MTCNN(keep_all=True, device='cuda')  ###This is a GPU optimized version, much faster on supported hardware
#from mtcnn import MTCNN
#detector = MTCNN()
from facenet_pytorch import MTCNN, InceptionResnetV1
resnet = InceptionResnetV1(pretrained='vggface2').eval()
#detector = MTCNN(keep_all=True, device='cuda')  ###This is a GPU optimized version, much faster on supported hardware
detector = MTCNN(keep_all=True)
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from time import sleep
from multiprocessing import Pool
import openpyxl
import imutils
from scipy.ndimage import rotate
import pickle
import importlib
import sys
import os

###In case you want to use the Github version of ISR
sys.path.append(os.path.abspath("~/GitHub/image-super-resolution/"))

from ISR.models import RDN, RRDN
rdn = RDN(weights='psnr-large')
rrdn = RRDN(weights='gans')

#import os
#import tensorflow as tf
#import tensorflow_hub as hub
#os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

#model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

####Avoid unecessary warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn


#####################################
####CORE MRZ Tessaract Functions#####
#####################################

####Birthday calculations (used to check if DOB and DOE are valid on passport extracted text
def calculateAge(birthDatestring):
	birthDate=datetime.strptime(birthDatestring, "%m/%d/%Y")
	today = date.today()
	age = today.year - birthDate.year - ((today.month, today.day) < (birthDate.month, birthDate.day))
	return age

####Core Tesseract implementation function, uses ORCB and runs checks to make sure data is valid
def readMRZ(image_path):
    mrz = read_mrz(image_path, extra_cmdline_params='--oem 0')
    mrz_dict = mrz.to_dict()
    mrz_frame = pd.DataFrame.from_dict(mrz_dict, orient='index').transpose()
    mrz_frame["Date of Birth"] = datetime.strptime(mrz_frame["date_of_birth"].astype("string")[0], "%y%m%d").strftime("%m/%d/%Y")
    mrz_frame["Date of Expiry"] = datetime.strptime(mrz_frame["expiration_date"].astype("string")[0], "%y%m%d").strftime("%m/%d/%Y")
    mrz_frame["MRZ number"] = mrz_frame["number"][0].replace("0", "O")
    if calculateAge(mrz_frame["Date of Birth"].astype("string")[0])<0:
        mrz_frame["Date of Birth"] = datetime.strftime(datetime.strptime(mrz_frame["Date of Birth"].astype("string")[0], "%m/%d/%Y") - relativedelta(years=100), "%m/%d/%Y")
    mrz_frame = mrz_frame.add_prefix('MRZ ')
    return mrz_frame

#####################################
###########Loading Data##############
#####################################

####Data loading and AI upscaling. These functions check if the file is a PDF or image and process accordingly.

###Passport upscaling
def preprocess_image(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def preprocess_image_online(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.image.decode_image(requests.get(image_path).content)
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save.
  """
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s.jpg" % filename)
  print("Saved as %s.jpg" % filename)

def passportUpscale(im_tensor):
    upscale_image = model(im_tensor)
    upscale_image = tf.squeeze(upscale_image)
    if not isinstance(upscale_image, Image.Image):
        image = tf.clip_by_value(upscale_image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    else:
        image = upscale_image
    return image


def passportByteStream(image_path):
    extension = "." + image_path.split(".", 2)[1].upper()
    if extension==".PDF":
        pages = convert_from_path(image_path)[0]
        with BytesIO() as f:
            pages.save(f)
            f.seek(0)
            img_page = cv2.imread(f)
    else:
        img_page=cv2.imread(image_path)
    return img_page

def passportCV(image_path, temp_path="~/"):
    extension = "." + image_path.split(".", 2)[1].upper()
    file = image_path.split(".", 2)[0]
    if extension==".PDF":
        page = convert_from_path(image_path)[0]
        page.save(temp_path + "_temp.jpeg", "JPEG")
        img = cv2.imread(temp_path + "_temp.jpeg")
    else:
        img=cv2.imread(image_path)
    return img

def passportCVUpscale(image_path, temp_path="~/", name="temp"):
    extension = "." + image_path.split(".", 2)[1].upper()
    file = image_path.split(".", 2)[0]
    if extension==".PDF":
        page = convert_from_path(image_path)[0]
        page.save(temp_path + "_temp.jpeg", "JPEG")
        img = preprocess_image(temp_path + "_temp.jpeg")
    else:
        img=preprocess_image(image_path)
    up_im = passportUpscale(img)
    save_image(up_im, temp_path + name + "_Final")
    new_img = cv2.imread(temp_path + name + "_Final.jpg")
    return new_img

####These are URL veresions of the above functions
def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    # return the image
    return image

def fileExt( url ):
    # compile regular expressions
    reQuery = re.compile( r'\?.*$', re.IGNORECASE )
    rePort = re.compile( r':[0-9]+', re.IGNORECASE )
    reExt = re.compile( r'(\.[A-Za-z0-9]+$)', re.IGNORECASE )
    # remove query string
    url = reQuery.sub( "", url )
    # remove port
    url = rePort.sub( "", url )
    # extract extension
    matches = reExt.search( url )
    if None != matches:
        return matches.group( 1 )
    return None

def passportCVOnline(image_path, temp_path):
    extension = fileExt(image_path).upper()
    #extension = "." + image_path.split(".", 2)[2].upper()
    file = image_path.split(".", 2)[0]
    if extension==".PDF":
        resp = requests.get(image_path)
        page = convert_from_bytes(resp.content)[0]
        page.save(temp_path + "temp.jpeg", "JPEG")
        img = cv2.imread(temp_path + "temp.jpeg")
    else:
        resp = urlopen(image_path)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return img

def passportCVUpscaleOnline(image_path, temp_path="~/", name="temp"):
    extension = fileExt(image_path).upper()
    file = image_path.split(".", 2)[0]
    if extension==".PDF":
        resp = requests.get(image_path)
        page = convert_from_bytes(resp.content)[0]
        page.save(temp_path + "_temp.jpeg", "JPEG")
        img = preprocess_image(temp_path + "_temp.jpeg")
    else:
        img=preprocess_image_online(image_path)
    up_im = passportUpscale(img)
    save_image(up_im, temp_path + name + "_Final")
    new_img = cv2.imread(temp_path + name + "_Final.jpg")
    return new_img


####Legacy image upscaling, depends on tensorflow 2.0 and ISR

def passportUpscale(data):
    if data.nbytes < 18000000:
        sr_img = rrdn.predict(data)
    else:
        sr_img = data
    #sr_img = rdn.predict(sr_img)
    return(sr_img)

def passportUpscaleFile(filepath, temp_path="~/"):
    data = passportCV(image_path=filepath, temp_path=temp_path)
    data_upscale = passportUpscale(data)
    data_img = Image.fromarray(cv2.cvtColor(data_upscale, cv2.COLOR_BGR2RGB))
    return(data_img)

def passportUpscaleNative(data):
    data_upscale = passportUpscale(data)
    data_img = Image.fromarray(cv2.cvtColor(data_upscale, cv2.COLOR_BGR2RGB))
    return data_img

def makePassportPrettyOnline(image_path, temp_path, brightness=35, contrast=11, left=0.36, top=0.78, right=4.26, bottom=1.96, delta=0.6, name="temp"):
    img = apply_brightness_contrast(correct_skew(passportCV(image_path=image_path, temp_path=temp_path), delta=delta, limit=1)[1], brightness, contrast)
    img1 = faceIDFull(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
    rect = deleteBlackBorder(img1)
    im = passportUpscaleNative(rect)
    return(im)



import tensorflow as tf

import os
import math
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.preprocessing import image_dataset_from_directory



#####################################
########Image Processing#############
#####################################
    
####Simple Fast Skew Correct
def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)
    return best_angle, rotated



####Much more complicated and slower skew correct
debug = True

#Display image
def display(img, frameName="OpenCV Image"):
    if not debug:
        return
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww*(h/w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
    cv2.waitKey(0)

#rotate the image with given theta value
def rotate_legacy(img, theta):
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols/2, rows/2)
    M = cv2.getRotationMatrix2D(image_center,theta,1)
    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])
    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)
    M[0, 2] += bound_w/2 - image_center[0]
    M[1, 2] += bound_h/2 - image_center[1]
    # rotate orignal image to show transformation
    rotated = cv2.warpAffine(img,M,(bound_w,bound_h),borderValue=(255,255,255))
    return rotated


def slope(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    slope = (y2-y1)/(x2-x1)
    theta = np.rad2deg(np.arctan(slope))
    return theta


def main(filePath):
    img = cv2.imread(filePath)
    textImg = img.copy()
    small = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)
    #find the gradient map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    display(grad)
    #Binarize the gradient image
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    display(bw)
    #connect horizontally oriented regions
    #kernal value (9,1) can be changed to improved the text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    display(connected)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    # _ , contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #opencv >= 4.0
    mask = np.zeros(bw.shape, dtype=np.uint8)
    #display(mask)
    #cumulative theta value
    cummTheta = 0
    #number of detected text regions
    ct = 0
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        #fill the contour
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        #display(mask)
        #ratio of non-zero pixels in the filled region
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        #assume at least 45% of the area is filled if it contains text
        if r > 0.45 and w > 8 and h > 8:
            #cv2.rectangle(textImg, (x1, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            rect = cv2.minAreaRect(contours[idx])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(textImg,[box],0,(0,0,255),2)
            #we can filter theta as outlier based on other theta values
            #this will help in excluding the rare text region with different orientation from ususla value
            theta = slope(box[0][0], box[0][1], box[1][0], box[1][1])
            cummTheta += theta
            ct +=1
            #print("Theta", theta)
    #find the average of all cumulative theta value
    orientation = cummTheta/ct
    print("Image orientation in degress: ", orientation)
    finalImage = rotate_legacy(img, orientation)
    display(textImg, "Detectd Text minimum bounding box")
    display(finalImage, "Deskewed Image")

def mainOnline(image_path, temp_path):
    img = passportCVOnline(image_path=image_path, temp_path=temp_path)
    textImg = img.copy()
    small = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)
    #find the gradient map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    display(grad)
    #Binarize the gradient image
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    display(bw)
    #connect horizontally oriented regions
    #kernal value (9,1) can be changed to improved the text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    display(connected)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    # _ , contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #opencv >= 4.0
    mask = np.zeros(bw.shape, dtype=np.uint8)
    #display(mask)
    #cumulative theta value
    cummTheta = 0
    #number of detected text regions
    ct = 0
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        #fill the contour
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        #display(mask)
        #ratio of non-zero pixels in the filled region
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        #assume at least 45% of the area is filled if it contains text
        if r > 0.45 and w > 8 and h > 8:
            #cv2.rectangle(textImg, (x1, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            rect = cv2.minAreaRect(contours[idx])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(textImg,[box],0,(0,0,255),2)
            #we can filter theta as outlier based on other theta values
            #this will help in excluding the rare text region with different orientation from ususla value
            theta = slope(box[0][0], box[0][1], box[1][0], box[1][1])
            cummTheta += theta
            ct +=1
            #print("Theta", theta)
    #find the average of all cumulative theta value
    orientation = cummTheta/ct
    print("Image orientation in degress: ", orientation)
    finalImage = rotate_legacy(img, orientation)
    display(textImg, "Detectd Text minimum bounding box")
    display(finalImage, "Deskewed Image")


#####Identify the largest rectangle (hopefully a passport page)
def biggestRectangle(contours):
    biggest = None
    max_area = 0
    indexReturn = -1
    for index in range(len(contours)):
        i = contours[index]
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.1 * peri, True)
            if area > max_area:  # and len(approx)==4:
                biggest = approx
                max_area = area
                indexReturn = index
    return indexReturn

###Adjust contrast and brightness, this is a core function used
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

####Trim White and Black borders
def deleteWhiteBorder(array):
    try:
        bw_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_array = 255*(bw_array < 128).astype(np.uint8)
        coords = cv2.findNonZero(white_array)
        x, y, w, h = cv2.boundingRect(coords)
        if h < 0:
            h=0
        if w < 0:
            w=0
        rect = array[y:y+h, x:x+w]
        #rect = cv2.bitwise_not(rect)
    except:
        rect = array
    return rect

def deleteBlackBorder(array):
    try:
        bw_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_array = 255*(bw_array < 128).astype(np.uint8)
        black_array = cv2.bitwise_not(white_array)
        coords = cv2.findNonZero(black_array)
        x, y, w, h = cv2.boundingRect(coords)
        rect = array[y:y+h, x:x+w]
        #rect = cv2.bitwise_not(rect)
    except:
        rect = array
    return rect

####Attempt to remove uncontoured (e.g. all white or all black) surroundings.
def deleteUnContouredBorder(array):
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(array,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = array[y:y+h,x:x+w]
    return crop

####Crop out large rectangles
def detectRectangleCrop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gray,kernel,iterations = 2)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 2)
    edged = cv2.Canny(dilation, 30, 200)
    _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    rects = sorted(rects,key=lambda  x:x[1],reverse=True)
    i = -1
    j = 1
    y_old = 5000
    x_old = 5000
    for rect in rects:
        x,y,w,h = rect
        area = w * h
        if area > 47000 and area < 70000:
            if (y_old - y) > 200:
                i += 1
                y_old = y
        if abs(x_old - x) > 300:
            x_old = x
            x,y,w,h = rect
            out = img[y+10:y+h-10,x+10:x+w-10]
    return rects[0]


#####################################
########Face Detection###############
#####################################

####This implementation uses pytorch and the MTCNN trained network (specifically InceptionResNetV1 and VGG. Code for both GPU (e.g. cuda) and cpu implementation are provided, we default to cpu here.


def mtcnnResults(data):
    result = detector.detect_faces(data)[0]
    boxes = np.expand_dims(np.array(result['box']), axis=0)
    boxes[0][2] = boxes[0][0] + boxes[0][2]
    boxes[0][3] = boxes[0][1] + boxes[0][3]
    probs = np.expand_dims(result['confidence'], axis=0)
    landmarks = np.expand_dims(np.array(list(result['keypoints'].values()),  dtype="float32"), axis=0)
    return boxes, probs, landmarks

####Face Recogniton check
def draw_image_with_boxes(filename):
    # load the image
    data = correct_skew(passportCV(filename), delta=0.5, limit=1)[1]
    boxes, probs, landmarks = detector.detect(data, landmarks=True)
    #boxes, probs, landmarks = mtcnnResults(data)
    # plot the image
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(data)
    ax.axis('off')
    passport_dim = boxes[0]
    vert = passport_dim[2] - passport_dim[0]
    horz = passport_dim[3] - passport_dim[1]
    passport_dim[0] = passport_dim[0]-(horz-(horz*0.54))
    if passport_dim[0] < 0:
        passport_dim[0]=0
    passport_dim[1] = passport_dim[1]-(vert+(vert*0.875))
    if passport_dim[1] < 0:
        passport_dim[1]=0
    passport_dim[2] = passport_dim[2]+horz*4.2
    if passport_dim[2] > data.shape[1]:
        passport_dim[2]=data.shape[1]
    passport_dim[3] = passport_dim[3]+vert*1.8
    if passport_dim[3] > data.shape[0]:
        passport_dim[3]=data.shape[0]
    ax.scatter(*np.meshgrid(passport_dim[[0, 2]], passport_dim[[1, 3]]), edgecolors='red')
    for box, landmark in zip(boxes, landmarks):
        ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
        ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
    fig.show()

####Same as above, but you can put a numpy darray instead of a filepath
def draw_image_with_boxes_debug(data):
    # load the image
    #boxes, probs, landmarks = mtcnnResults(data)
    boxes, probs, landmarks = detector.detect(data, landmarks=True)
    # plot the image
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(data)
    ax.axis('off')
    passport_dim = boxes[0]
    vert = passport_dim[2] - passport_dim[0]
    horz = passport_dim[3] - passport_dim[1]
    passport_dim[0] = passport_dim[0]-(horz-(horz*0.54))
    if passport_dim[0] < 0:
        passport_dim[0]=0
    passport_dim[1] = passport_dim[1]-(vert+(vert*0.875))
    if passport_dim[1] < 0:
        passport_dim[1]=0
    passport_dim[2] = passport_dim[2]+horz*4.2
    if passport_dim[2] > data.shape[1]:
        passport_dim[2]=data.shape[1]
    passport_dim[3] = passport_dim[3]+vert*1.8
    if passport_dim[3] > data.shape[0]:
        passport_dim[3]=data.shape[0]
    ax.scatter(*np.meshgrid(passport_dim[[0, 2]], passport_dim[[1, 3]]), edgecolors='red')
    for box, landmark in zip(boxes, landmarks):
        ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
        ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
    fig.show()



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def riseRun(coord_1, coord_2):
    d_x = coord_2[0]-coord_1[0]
    d_y = coord_2[1]-coord_1[1]
    return d_y/d_x

def faceSlope(landmarks):
    landmarks = NormalizeData(landmarks)
    coord_2 = landmarks[0][1,]
    coord_1 = landmarks[0][3,]
    return riseRun(coord_1, coord_2)

def slopeAngle(m1, m2):
    numerator = m2 - m1
    denominator = 1 + (m1*m2)
    return math.degrees(numerator/denominator)

def faceDegrees(landmarks):
    m1 = -1.2281231
    m2 = faceSlope(landmarks)
    return slopeAngle(m1, m2)
    
    
####Face ID detection. This is the basic function. It works by detecitng the face in the data, measures its height and width, and then adds multipliers (left, top, right, and bottom) to generate internall measures of passport dimensions.
def faceIDSimple(data, left=0.64, top=0.875, right=4.2, bottom=2.5, delta=0.5):
    #boxes, probs, landmarks = mtcnnResults(data)
    boxes, probs, landmarks = detector.detect(data, landmarks=True)
    throw = len(boxes)
    passport_dim = boxes[0]
    vert = passport_dim[2] - passport_dim[0]
    horz = passport_dim[3] - passport_dim[1]
    passport_dim[0] = np.round(passport_dim[0]-(horz-(horz*left)), 0)
    if passport_dim[0] < 0:
        passport_dim[0]=0
    passport_dim[1] = np.round(passport_dim[1]-(vert+(vert*top)), 0)
    if passport_dim[1] < 0:
        passport_dim[1]=0
    passport_dim[2] = np.round(passport_dim[2]+horz*right, 0)
    if passport_dim[2] > data.shape[1]:
        passport_dim[2]=data.shape[1]
    passport_dim[3] = np.round(passport_dim[3]+vert*bottom, 0)
    if passport_dim[3] > data.shape[0]:
        passport_dim[3]=data.shape[0]
    passport_dim = passport_dim.astype(int)
    #im1 = data.crop(passport_dim)
    im1 = data[passport_dim[1]:passport_dim[3], passport_dim[0]:passport_dim[2]]
    return(im1)

def faceIDPre(data, left=0.64, top=0.875, right=4.2, bottom=2.5, delta=0.5):
    #boxes, probs, landmarks = mtcnnResults(data)
    boxes, probs, landmarks = detector.detect(data, landmarks=True)
    data = rotate(data, angle=faceDegrees(landmarks))
    data = correct_skew(data, delta=delta, limit=1)[1]
    boxes, probs, landmarks = detector.detect(data, landmarks=True)
    #boxes, probs, landmarks = mtcnnResults(data)
    throw = len(boxes)
    passport_dim = boxes[0]
    vert = passport_dim[2] - passport_dim[0]
    horz = passport_dim[3] - passport_dim[1]
    passport_dim[0] = np.round(passport_dim[0]-(horz-(horz*left)), 0)
    if passport_dim[0] < 0:
        passport_dim[0]=0
    passport_dim[1] = np.round(passport_dim[1]-(vert+(vert*top)), 0)
    if passport_dim[1] < 0:
        passport_dim[1]=0
    passport_dim[2] = np.round(passport_dim[2]+horz*right, 0)
    if passport_dim[2] > data.shape[1]:
        passport_dim[2]=data.shape[1]
    passport_dim[3] = np.round(passport_dim[3]+vert*bottom, 0)
    if passport_dim[3] > data.shape[0]:
        passport_dim[3]=data.shape[0]
    passport_dim = passport_dim.astype(int)
    #im1 = data.crop(passport_dim)
    im1 = data[passport_dim[1]:passport_dim[3], passport_dim[0]:passport_dim[2]]
    return(im1)

###This checks the faceID function against multiple image orientations
def faceIDFull(data, left=0.64, top=0.875, right=4.2, bottom=2.5, delta=0.5):
    try:
        new_data = correct_skew(data, delta=0.5, limit=1)[1]
        result = faceIDPre(new_data, left=left, top=top, right=right, bottom=bottom, delta=delta)
        return result
    except:
        try:
            data90 = np.rot90(data, k=1)
            data90 = correct_skew(data90, delta=0.5, limit=1)[1]
            result = faceIDPre(data90, left=left, top=top, right=right, bottom=bottom, delta=delta)
            return result
        except:
            try:
                data270 = np.rot90(data, k=3)
                data270 = correct_skew(data180, delta=0.5, limit=1)[1]
                result = faceIDPre(data270, left=left, top=top, right=right, bottom=bottom, delta=delta)
                return result
            except:
                try:
                    data180 = np.rot90(data, k=2)
                    data180 = correct_skew(data180, delta=0.5, limit=1)[1]
                    result = faceIDPre(data180, left=left, top=top, right=right, bottom=bottom, delta=delta)
                    return result
                except:
                    pass

##Default faceID function, takes image and on failure rotates 90 degrees.
def faceID(data, left=0.64, top=0.875, right=4.2, bottom=2.5, delta=0.5):
    try:
        new_data = correct_skew(data, delta=0.5, limit=1)[1]
        result = faceIDPre(new_data, left=left, top=top, right=right, bottom=bottom, delta=delta)
        return result
    except:
        data90 = np.rot90(data, k=1)
        data90 = correct_skew(data90, delta=0.5, limit=1)[1]
        result = faceIDPre(data90, left=left, top=top, right=right, bottom=bottom, delta=delta)
        return result

###Simple combination of facial recognition and tesseract
def readMRZCrop(image_path):
    file = image_path.split(".", 2)[0]
    img = apply_brightness_contrast(correct_skew(passportCV(image_path), delta=1, limit=1)[1], -25, -29)
    #img = correct_skew(img, delta=0.5, limit=1)[1]
    #img = correct_skew(img)
    #img = passportCV(image_path)
    img1 = faceID(img)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #rect = deleteWhiteBorder(img1)
    rect = deleteBlackBorder(img1)
    #rect = deleteWhiteBorder(rect)
    #rect = deleteBlackBorder(rect)
    rect = passportUpscale(rect)
    im = Image.fromarray(rect)
    im.save(file + "_temp.jpeg", "JPEG")
    result = readMRZ(file + "_temp.jpeg")
    return result


###Local file combination of facial recogniton and tesseract. Note that this will atempt once, flip upsidedown, then continue.
def readMRZCropFile(image_path, temp_path, brightness=-27, contrast=-32, left=0.64, top=0.875, right=4.2, bottom=2.5, delta=0.5, name="temp"):
    img = apply_brightness_contrast(correct_skew(passportCV(image_path=image_path, temp_path=temp_path), delta=delta, limit=1)[1], brightness, contrast)
    try:
        try:
            img1 = faceIDSimple(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
            img1 = correct_skew(img1, delta=delta, limit=1)[1]
            rect = deleteBlackBorder(img1)
            im = Image.fromarray(rect)
            im.save(temp_path + name + "_Final.jpeg", "JPEG")
            result = readMRZ(temp_path + name + "_Final.jpeg")
            rect = passportUpscale(img)
            im = Image.fromarray(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
            im.save(temp_path + name + "_Final.jpeg", "JPEG")
            return result
        except:
            img = np.rot90(img, k=2)
            img1 = faceIDFull(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
            img1 = correct_skew(img1, delta=delta, limit=1)[1]
            rect = deleteBlackBorder(img1)
            im = Image.fromarray(rect)
            im.save(temp_path + name + "_Final.jpeg", "JPEG")
            result = readMRZ(temp_path + name + "_Final.jpeg")
            rect = passportUpscale(img)
            im = Image.fromarray(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
            im.save(temp_path + name + "_Final.jpeg", "JPEG")
            return result
    except:
        rect1 = passportUpscale(img)
        im = Image.fromarray(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
        im.save(temp_path + name + "_Final.jpeg", "JPEG")

###URL fetching combination of facial recogniton and tesseract. Note that this will atempt once, flip upsidedown, then continue.
def readMRZCropOnline(image_path, temp_path, brightness=-27, contrast=-32, left=0.68, top=0.52, right=4.56, bottom=1.8, delta=0.25, name="temp"):
    img = apply_brightness_contrast(correct_skew(passportCVOnline(image_path=image_path, temp_path=temp_path), delta=delta, limit=1)[1], brightness, contrast)
    height = bottom - top
    width = right - left
    
    try:
        try:
            img1 = faceIDSimple(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
            rect = deleteBlackBorder(img1)
            #rect = passportUpscale(rect)
            im = Image.fromarray(rect)
            im.save(temp_path + name + "_Final.jpeg", "JPEG")
            result = readMRZ(temp_path + name + "_Final.jpeg")
            rect = passportUpscale(img)
            im = Image.fromarray(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
            im.save(temp_path + name + "_Final.jpeg", "JPEG")
            return result
        except:
            img = np.rot90(img, k=2)
            img1 = faceIDFull(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
            rect = deleteBlackBorder(img1)
            im = Image.fromarray(rect)
            im.save(temp_path + name + "_Final.jpeg", "JPEG")
            result = readMRZ(temp_path + name + "_Final.jpeg")
            rect = passportUpscale(img)
            im = Image.fromarray(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
            im.save(temp_path + name + "_Final.jpeg", "JPEG")
            return result
    except:
        rect = passportUpscale(img)
        im = Image.fromarray(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
        im.save(temp_path + name + "_Final.jpeg", "JPEG")
        
def readMRZCropOnlineRotate(image_path, temp_path, brightness=-27, contrast=-32, left=0.64, top=0.875, right=4.2, bottom=2.5, delta=0.5, name="temp"):
    img = apply_brightness_contrast(correct_skew(passportCVOnline(image_path=image_path, temp_path=temp_path), delta=delta, limit=1)[1], brightness, contrast)
    height = bottom - top
    width = right - left
    try:
        try:
            img1 = faceIDSimple(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
            rect = deleteBlackBorder(img1)
            #rect = passportUpscale(rect)
            im = Image.fromarray(rect)
            im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
            im.save(temp_path + name + "_Final.jpeg", "JPEG")
            result = readMRZ(temp_path + name + "_Final.jpeg")
            return result
        except:
            try:
                img1 = faceIDSimple(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
                rect = deleteBlackBorder(img1)
                rect = deleteWhiteBorder(rect)
                im = Image.fromarray(rect)
                im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                im.save(temp_path + name + "_Final.jpeg", "JPEG")
                result = readMRZ(temp_path + name + "_Final.jpeg")
                return result
            except:
                try:
                    img1 = faceIDSimple(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
                    rect = deleteBlackBorder(img1)
                    rect = deleteWhiteBorder(rect)
                    rect = deleteBlackBorder(rect)
                    im = Image.fromarray(rect)
                    im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                    im.save(temp_path + name + "_Final.jpeg", "JPEG")
                    result = readMRZ(temp_path + name + "_Final.jpeg")
                    return result
                except:
                    img = np.rot90(img, k=2)
                    img1 = faceIDFull(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
                    rect = deleteBlackBorder(img1)
                    im = Image.fromarray(rect)
                    im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                    im.save(temp_path + name + "_Final.jpeg", "JPEG")
                    result = readMRZ(temp_path + name + "_Final.jpeg")
                    return result
    except:
        pass

def readMRZCropOnlineSimple(image_path, temp_path, brightness=-27, contrast=-32, left=0.64, top=0.875, right=4.2, bottom=2.5, delta=0.5, name="temp"):
    img = apply_brightness_contrast(correct_skew(passportCVOnline(image_path=image_path, temp_path=temp_path), delta=delta, limit=1)[1], brightness, contrast)
    try:
        try:
            img1 = faceIDSimple(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
            rect = deleteBlackBorder(img1)
            #rect = passportUpscale(rect)
            im = Image.fromarray(rect)
            im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
            im.save(temp_path + name + "_Final.jpeg", "JPEG")
            result = readMRZ(temp_path + name + "_Final.jpeg")
            return result
        except:
            try:
                img1 = faceIDSimple(img, left=left, top=top, right=right, bottom=bottom-0.2, delta=delta)
                rect = deleteBlackBorder(img1)
                im = Image.fromarray(rect)
                im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                im.save(temp_path + name + "_Final.jpeg", "JPEG")
                result = readMRZ(temp_path + name + "_Final.jpeg")
                return result
            except:
                try:
                    img1 = faceIDSimple(img, left=left, top=top, right=right, bottom=bottom-0.4, delta=delta)
                    rect = deleteBlackBorder(img1)
                    im = Image.fromarray(rect)
                    im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                    im.save(temp_path + name + "_Final.jpeg", "JPEG")
                    result = readMRZ(temp_path + name + "_Final.jpeg")
                    return result
                except:
                    try:
                        img1 = faceIDSimple(img, left=left, top=top, right=right, bottom=bottom-0.6, delta=delta)
                        rect = deleteBlackBorder(img1)
                        im = Image.fromarray(rect)
                        im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                        im.save(temp_path + name + "_Final.jpeg", "JPEG")
                        result = readMRZ(temp_path + name + "_Final.jpeg")
                        return result
                    except:
                        try:
                            img1 = faceIDSimple(img, left=left-0.2, top=top, right=right, bottom=bottom, delta=delta)
                            rect = deleteBlackBorder(img1)
                            #rect = passportUpscale(rect)
                            im = Image.fromarray(rect)
                            im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                            im.save(temp_path + name + "_Final.jpeg", "JPEG")
                            result = readMRZ(temp_path + name + "_Final.jpeg")
                            return result
                        except:
                            try:
                                img1 = faceIDSimple(img, left=left-0.2, top=top, right=right, bottom=bottom-0.2, delta=delta)
                                rect = deleteBlackBorder(img1)
                                im = Image.fromarray(rect)
                                im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                                im.save(temp_path + name + "_Final.jpeg", "JPEG")
                                result = readMRZ(temp_path + name + "_Final.jpeg")
                                return result
                            except:
                                try:
                                    img1 = faceIDSimple(img, left=left-0.2, top=top, right=right, bottom=bottom-0.4, delta=delta)
                                    rect = deleteBlackBorder(img1)
                                    im = Image.fromarray(rect)
                                    im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                                    im.save(temp_path + name + "_Final.jpeg", "JPEG")
                                    result = readMRZ(temp_path + name + "_Final.jpeg")
                                    return result
                                except:
                                    try:
                                        img1 = faceIDSimple(img, left=left-0.2, top=top, right=right, bottom=bottom-0.6, delta=delta)
                                        rect = deleteBlackBorder(img1)
                                        im = Image.fromarray(rect)
                                        im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                                        im.save(temp_path + name + "_Final.jpeg", "JPEG")
                                        result = readMRZ(temp_path + name + "_Final.jpeg")
                                        return result
                                    except:
                                        try:
                                            img1 = faceIDSimple(img, left=left-0.4, top=top, right=right, bottom=bottom, delta=delta)
                                            rect = deleteBlackBorder(img1)
                                            #rect = passportUpscale(rect)
                                            im = Image.fromarray(rect)
                                            im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                                            im.save(temp_path + name + "_Final.jpeg", "JPEG")
                                            result = readMRZ(temp_path + name + "_Final.jpeg")
                                            return result
                                        except:
                                            try:
                                                img1 = faceIDSimple(img, left=left-0.4, top=top, right=right, bottom=bottom-0.2, delta=delta)
                                                rect = deleteBlackBorder(img1)
                                                im = Image.fromarray(rect)
                                                im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                                                im.save(temp_path + name + "_Final.jpeg", "JPEG")
                                                result = readMRZ(temp_path + name + "_Final.jpeg")
                                                return result
                                            except:
                                                try:
                                                    img1 = faceIDSimple(img, left=left-0.4, top=top, right=right, bottom=bottom-0.4, delta=delta)
                                                    rect = deleteBlackBorder(img1)
                                                    im = Image.fromarray(rect)
                                                    im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                                                    im.save(temp_path + name + "_Final.jpeg", "JPEG")
                                                    result = readMRZ(temp_path + name + "_Final.jpeg")
                                                    return result
                                                except:
                                                    img1 = faceIDSimple(img, left=left-0.4, top=top, right=right, bottom=bottom-0.6, delta=delta)
                                                    rect = deleteBlackBorder(img1)
                                                    im = Image.fromarray(rect)
                                                    im = im.resize((1020, math.ceil(1020*(im.size[1]/im.size[0]))), Image.ANTIALIAS)
                                                    im.save(temp_path + name + "_Final.jpeg", "JPEG")
                                                    result = readMRZ(temp_path + name + "_Final.jpeg")
                                                    return result
    except:
        pass

###Numpy darray combination of facial recogniton and tesseract. Note that this will atempt once, flip upsidedown, then continue. This assumes images have already been loaded
def readMRZCropNative(image_object, temp_path, brightness=-27, contrast=-32, left=0.64, top=0.875, right=4.2, bottom=2.5, delta=0.5, name="temp"):
    img = apply_brightness_contrast(correct_skew(image_object, delta=delta, limit=1)[1], brightness, contrast)
    try:
        img1 = faceIDSimple(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
        rect = deleteBlackBorder(img1)
        rect = passportUpscale(rect)
        im = Image.fromarray(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
        im.save(temp_path + name + ".jpeg", "JPEG")
        result = readMRZ(temp_path + name + ".jpeg")
        return result
    except:
        img = np.rot90(img, k=2)
        img1 = faceIDFull(img, left=left, top=top, right=right, bottom=bottom, delta=delta)
        rect = deleteBlackBorder(img1)
        rect = passportUpscale(rect)
        im = Image.fromarray(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
        im.save(temp_path + name + ".jpeg", "JPEG")
        result = readMRZ(temp_path + name + ".jpeg")
        return result


###This function assumes passport urls are in a spreadsheet, it will attempt to update the spreadsheet with verified information
def passportSheetEvaluate(worksheet, row, temp_path):
    worksheet_sub = worksheet.iloc[row,]
    try:
        passport_text = readMRZCropOnline(image_path=worksheet_sub["Passport Link"], temp_path="~/Desktop/Passport Training/")
        worksheet_sub["Verified"] = "Yes"
        worksheet_sub["AIBirthday"] = passport_text["Date of Birth"][0]
        worksheet_sub["AIExpiry"] = passport_text["Date of Expiry"][0]
        worksheet_sub["AIExpiry"] = passport_text["Date of Expiry"][0]
        worksheet_sub["AISurname"] = passport_text["surname"][0]
    except:
        pass
    return worksheet_sub


####I really like this progress bar for for loops.
def printProgressBar(i,max,postText):
    n_bar =100 #size of progress bar
    j= i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()



###Not great alternatives to the above functions, but providing in case it is useful.
def readMRZCropContour(image_path):
    file = image_path.split(".", 2)[0]
    img = apply_brightness_contrast(passportCV(image_path), -60, 60)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #invGamma = 1.0 / 0.3
    #table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    #gray = cv2.LUT(gray, table)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 128).astype(np.uint8)
    ret, thresh1 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    indexReturn = biggestRectangle(contours)
    hull = cv2.convexHull(contours[indexReturn])
    mask = np.zeros_like(img)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, contours, indexReturn, 255, -1)  # Draw filled contour in mask
    out = np.zeros_like(img)  # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]
    # crop the image
    (y, x, _) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = img[topy : bottomy + 1, topx : bottomx + 1, :]
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(out_rgb)
    im.save(file + "_temp.jpeg", "JPEG")
    result = readMRZ(file + "_temp.jpeg")
    return result


def readMRZCropBorders(image_path):
    file = image_path.split(".", 2)[0]
    img = apply_brightness_contrast(passportCV(image_path), -20, 0)
    white = [255,255,255]
    img = cv2.copyMakeBorder(img,60,60,60,60,cv2.BORDER_CONSTANT,value=white)
    # Convert from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #multiple by a factor to change the saturation
    hsv[...,1] = hsv[...,1]*1.6
    #multiple by a factor of less than 1 to reduce the brightness
    hsv[...,2] = hsv[...,2]*0.9
    #greenMask = cv2.inRange(hsv, (26, 10, 30), (97, 100, 255))
    #hsv[greenMask == 255] = (0, 255, 0)
    # Get the saturation plane - all black/white/gray pixels are zero, and colored pixels are above zero.
    s = hsv[:, :, 1]
    # Apply threshold on s - use automatic threshold algorithm (use THRESH_OTSU).
    ret, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    # Find the contour with the maximum area.
    c = max(cnts, key=cv2.contourArea)
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(c)
    # Crop the bounding rectangle out of img
    out = img[y:y+h, x:x+w, :].copy()
    im = Image.fromarray(out)
    im.save(file + "_temp.jpeg", "JPEG")
    result = readMRZ(file + "_temp.jpeg")
    return result







