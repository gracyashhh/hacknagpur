import cv2

#Deepface is a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python
from deepface import DeepFace

import matplotlib.pyplot as plt 

img = cv2.imread("/content/r7.jpg")

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

predictions = DeepFace.analyze(img)

print(predictions)

#Haarcascade xml file to be gotten from the link provided below
#https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'/content/haarcascade_frontalface_default.xml')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=faceCascade.detectMultiScale(gray,1.1,4)

for(x,y,w,h) in faces:
  cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
  
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,
            predictions['dominant_emotion'],
            (0,50),font,1,(255,0,0),2,cv2.LINE_4)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
