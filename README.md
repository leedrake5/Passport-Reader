# Passport Reader

This code is designed to help facilitate the reading of the MRZ codes of passports in the event that the passport image is less than perfect. It uses a combination of MTCNN and OpenCV to properly orient documents for scanning

![alt text](https://raw.githubusercontent.com/leedrake5/Passport-Reader/master/examples/IMG_0343.jpg)


## How it works
[Multi-task cascaded convolutional networks (MTCNN)](https://medium.com/@iselagradilla94/multi-task-cascaded-convolutional-networks-mtcnn-for-face-detection-and-facial-landmark-alignment-7c21e8007923#:~:text=Multi%2Dtask%20Cascaded%20Convolutional%20Networks%20(MTCNN)%20is%20a%20framework,eyes%2C%20nose%2C%20and%20mouth.) can both detect faces and identify key landmarks (eyes, nose, etc.). These can be used to properly orient an image. 

![alt text](https://raw.githubusercontent.com/leedrake5/Passport-Reader/master/examples/MTCNNexample.png)

These landmarks can be used to define a face. But because passport images are conservative (e.g. face orientation and dimensions is intended to be consistent) we can extraploate passport document boundaries based on these landmarks (red dots in the above figure). Note that we have also adjusted brightness and contrast to increase the contrast of text to backround to facilitate document reading. 

![alt text](https://raw.githubusercontent.com/leedrake5/Passport-Reader/master/examples/astley_skew.jpeg)

In the event a passport document is extremely skewed - e.g. the placement of the passport is uneven - we can use these MTCNN landmarks to properly orient the document to once again facilitate document processing. 

![alt text](https://raw.githubusercontent.com/leedrake5/Passport-Reader/master/examples/astley_temp.jpeg)

Just like that! Once the document is ready for reading, we can use [PassportEye](https://github.com/konstantint/PassportEye) to process the document and extract usable data to verify or generate information. This uses the MRZ code (bottom two lines of code at the bottom of the passport) to extract key information like passport number, date of birth, and date of expiry.

![alt text](https://raw.githubusercontent.com/leedrake5/Passport-Reader/master/examples/example_output.jpeg)
