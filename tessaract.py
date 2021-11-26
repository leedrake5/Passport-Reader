###Install ORCB
###Move the included orcb.traineddata file (in the data folder) to your tesseract/shared folder - this will be deep in system information

###The 4 passports provided with this repository are public access and do not include any private information that puts anyone at risk. They come from this website: https://www.consilium.europa.eu/prado/en/prado-documents/AFG/A/docs-per-category.html

###Most Functional, minimal version
###Note that this requires tesseract to be installed externally, and the ORCB library. 

from passporteye import read_mrz
import sys
import cv2
import numpy as np
import pytesseract
from datetime import datetime
import pandas as pd
import pytesseract
from PIL import Image

from datetime import date
from dateutil.relativedelta import relativedelta

def calculateAge(birthDatestring):
	birthDate=datetime.strptime(birthDatestring, "%m/%d/%Y")
	today = date.today()
	age = today.year - birthDate.year - ((today.month, today.day) < (birthDate.month, birthDate.day))
	return age

def readMRZ(image_path):
	mrz = read_mrz(image_path, extra_cmdline_params='orcb')
	mrz_dict = mrz.to_dict()
	mrz_frame = pd.DataFrame.from_dict(mrz_dict, orient='index').transpose()
	mrz_frame["Date of Birth"] = datetime.strptime(mrz_frame["date_of_birth"].astype("string")[0], "%y%m%d").strftime("%m/%d/%Y")
	mrz_frame["Date of Expiry"] = datetime.strptime(mrz_frame["expiration_date"].astype("string")[0], "%y%m%d").strftime("%m/%d/%Y")
	if calculateAge(mrz_frame["Date of Birth"].astype("string")[0])<0:
		mrz_frame["Date of Birth"] = datetime.strftime(datetime.strptime(mrz_frame["Date of Birth"].astype("string")[0], "%m/%d/%Y") - relativedelta(years=100), "%m/%d/%Y")
	return mrz_frame
	
	
first = readMRZ("examples/IMG_0343.jpg")
second = readMRZ("examples/IMG_0344.jpg")
third = readMRZ("examples/IMG_0345.jpg")
fourth = readMRZ("examples/IMG_0346.jpg")


###Other examples, though this is really scratch code that should be considered a platform for building new tools. Image editing by cv2 may provide more help, but ultimate problems are related to passport orientation in the photo. 
import sys
import cv2
import numpy as np
import pytesseract
from datetime import datetime
import pandas as pd

startTime = datetime.now()

input_image_path = "examples/IMG_0343.jpg"

img = cv2.imread(input_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
invGamma = 1.0 / 0.
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(
    "uint8"
)

# apply gamma correction using the lookup table
gray = cv2.LUT(gray, table)

ret, thresh1 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[
    -2:
]


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


indexReturn = biggestRectangle(contours)
hull = cv2.convexHull(contours[indexReturn])

# create a crop mask
mask = np.zeros_like(img)  # Create mask where white is what we want, black otherwise
cv2.drawContours(mask, contours, indexReturn, 255, -1)  # Draw filled contour in mask
out = np.zeros_like(img)  # Extract out the object and place into output image
out[mask == 255] = img[mask == 255]

# crop the image
(y, x, _) = np.where(mask == 255)
(topy, topx) = (np.min(y), np.min(x))
(bottomy, bottomx) = (np.max(y), np.max(x))
out = img[topy : bottomy + 1, topx : bottomx + 1, :]


# predict tesseract
lang = "eng+nld"
config = "--psm 11 --oem 3"
out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# uncomment to see raw prediction
# print(pytesseract.image_to_string(out_rgb, lang=lang, config=config))


img_data = pytesseract.image_to_data(
    out_rgb,
    lang=lang,
    config=config,
    output_type=pytesseract.Output.DATAFRAME,
)
img_conf_text = img_data[["conf", "text"]]
img_valid = img_conf_text[img_conf_text["text"].notnull()]
img_words = img_valid[img_valid["text"].str.len() > 1]

# to see confidence of one word
# word = "Gulfaraz"
# print(img_valid[img_valid["text"] == word])

all_predictions = img_words["text"].to_list()
print(all_predictions)

confidence_level = 90

img_conf = img_words[img_words["conf"] > confidence_level]
predictions = img_conf["text"].to_list()

# uncomment to see confident predictions
# print(predictions)

print("Execution Time: {}".format(datetime.now() - startTime))

###Google Drive Version

function doGet(request) {
  if (request.parameters.url != undefined && request.parameters.url != "") {
    var imageBlob = UrlFetchApp.fetch(request.parameters.url).getBlob();
    var resource = {
      title: imageBlob.getName(),
      mimeType: imageBlob.getContentType()
    };
    var options = {
      ocr: true
    };
    var docFile = Drive.Files.insert(resource, imageBlob, options);
    var doc = DocumentApp.openById(docFile.id);
    var text = doc.getBody().getText().replace("\n", "");
    Drive.Files.remove(docFile.id);
    return ContentService.createTextOutput(text);
  }
  else {
    return ContentService.createTextOutput("request error");
  }
}



#####Passport Eye Version
from passporteye import read_mrz
import sys
import cv2
import numpy as np
import pytesseract
from datetime import datetime
import pandas as pd
import pytesseract
from PIL import Image

startTime = datetime.now()

input_image_path = "examples/IMG_0344.jpg"

img = cv2.imread(input_image_path)
image_bytes = img_roi.tobytes(order='C')

im = Image.fromarray(img)
im.save("examples/IMG_0344_mod.jpg")

mrz=read_mrz("examples/IMG_0346.jpg", extra_cmdline_params='-1 orcb')
mrz_data = mrz.to_dict()

print(mrz_data['country'])
print(mrz_data['names'])
print(mrz_data['surname'])
print(mrz_data['type'])
print(mrz_data['date_of_birth'])
print(mrz_data['expiration_date'])

text = pytesseract.image_to_string(input_image_path, lang="orcb")


from passporteye.mrz.image import MRZPipeline 
p = MRZPipeline(input_image_path)
mrz = p.result

###Turn it into a function
from datetime import date
from dateutil.relativedelta import relativedelta

def calculateAge(birthDatestring):
	birthDate=datetime.strptime(birthDatestring, "%m/%d/%Y")
	today = date.today()
	age = today.year - birthDate.year - ((today.month, today.day) < (birthDate.month, birthDate.day))
	return age

def readMRZ(image_path):
	mrz = read_mrz(image_path, extra_cmdline_params='-1 orcb')
	mrz_dict = mrz.to_dict()
	mrz_frame = pd.DataFrame.from_dict(mrz_dict, orient='index').transpose()
	mrz_frame["Date of Birth"] = datetime.strptime(mrz_frame["date_of_birth"].astype("string")[0], "%y%m%d").strftime("%m/%d/%Y")
	mrz_frame["Date of Expiry"] = datetime.strptime(mrz_frame["expiration_date"].astype("string")[0], "%y%m%d").strftime("%m/%d/%Y")
	if calculateAge(mrz_frame["Date of Birth"].astype("string")[0])<0:
		mrz_frame["Date of Birth"] = datetime.strftime(datetime.strptime(mrz_frame["Date of Birth"].astype("string")[0], "%m/%d/%Y") - relativedelta(years=100), "%m/%d/%Y")
	return mrz_frame
	
	
second = readMRZ("examples/IMG_0343.jpg")

###Other Implementation
import cv2
import pytesseract

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

#####################################

img = cv2.imread('examples/IMG_0343.jpg',0)
(height, width) = img.shape

#####################################

img_copy = img.copy()

img_canny = cv2.Canny(img_copy, 50, 100, apertureSize = 3)

img_hough = cv2.HoughLinesP(img_canny, 1, math.pi / 180, 100, minLineLength = 100, maxLineGap = 10)

(x, y, w, h) = (np.amin(img_hough, axis = 0)[0,0], np.amin(img_hough, axis = 0)[0,1],
np.amax(img_hough, axis = 0)[0,0] - np.amin(img_hough, axis = 0)[0,0],
np.amax(img_hough, axis = 0)[0,1] - np.amin(img_hough, axis = 0)[0,1])

img_roi = img_copy[y:y+h,x:x+w]

#####################################

img_roi = cv2.rotate(img_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

(height, width) = img_roi.shape

img_roi_copy = img_roi.copy()
dim_mrz = (x, y, w, h) = (1, round(height*0.9), width-3, round(height-(height*0.9))-2)
img_roi_copy = cv2.rectangle(img_roi_copy, (x, y), (x + w ,y + h),(0,0,0),2)

img_mrz = img_roi[y:y+h, x:x+w]
img_mrz =cv2.GaussianBlur(img_mrz, (3,3), 0)
ret, img_mrz = cv2.threshold(img_mrz,127,255,cv2.THRESH_TOZERO)

mrz = pytesseract.image_to_string(img_mrz, config = '--psm 12')
mrz = [line for line in mrz.split('\n') if len(line)>10]

if mrz[0][0:2] == 'P<':
	lastname = mrz[0].split('<')[1][3:]
else:
	lastname = mrz[0].split('<')[0][5:]

firstname = [i for i in mrz[0].split('<') if (i).isspace() == 0 and len(i) > 0][1]

pp_no = mrz[1][:9]

###################################

img_roi_copy = img_roi.copy()
dim_lastname_chi = (x, y, w, h) = (455, 1210, 120, 70)
img_roi_copy = cv2.rectangle(img_roi_copy, (x, y), (x + w ,y + h),(0,0,0))

img_lastname_chi = img_roi[y:y+h, x:x+w]
img_lastname_chi = cv2.GaussianBlur(img_lastname_chi, (3,3), 0)
ret, img_lastname_chi = cv2.threshold(img_lastname_chi,127,255,cv2.THRESH_TOZERO)

lastname_chi = pytesseract.image_to_string(img_lastname_chi, lang = 'chi_sim', config = '--psm 7')
lastname_chi = lastname_chi.split('\n')[0]

dim_firstname_chi = (x, y, w, h) = (455, 1300, 120, 70)
img_roi_copy = cv2.rectangle(img_roi_copy, (x, y), (x + w ,y + h),(0,0,0))

img_firstname_chi = img_roi[y:y+h, x:x+w]
img_firstname_chi =cv2.GaussianBlur(img_firstname_chi, (3,3), 0)
ret, img_firstname_chi = cv2.threshold(img_firstname_chi,127,255,cv2.THRESH_TOZERO)

firstname_chi = pytesseract.image_to_string(img_firstname_chi, lang = 'chi_sim', config = '--psm 7')
firstname_chi = firstname_chi.split('\n')[0]

passport_dict = {'Passport No.': pp_no,
                 'First Name': firstname,
                 'Last Name': lastname,
                 'First Name (汉字)': firstname_chi,
                 'Last Name (汉字)': lastname_chi}

output = pd.DataFrame(columns = ['Passport No.','First Name','Last Name','First Name (汉字)','Last Name (汉字)'])
output = output.append(passport_dict, ignore_index = True)

output.to_excel("output.xlsx", index = False)









