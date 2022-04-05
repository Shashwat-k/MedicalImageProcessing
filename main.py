import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('Img/elon1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
testElon = face_recognition.load_image_file('Img/elon2.jpg')
testElon = cv2.cvtColor(testElon, cv2.COLOR_BGR2RGB)
testBill = face_recognition.load_image_file('Img/bill1.jpg')
testBill = cv2.cvtColor(testBill, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(155,25,10),2)

faceLocTest = face_recognition.face_locations(testElon)[0]
encodeElonTest = face_recognition.face_encodings(testElon)[0]
cv2.rectangle(testElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(155,25,10),2)

encodeBillTest = face_recognition.face_encodings(testBill)[0]

faceDis = face_recognition.face_distance([encodeElon],encodeElonTest)
faceDis1 = face_recognition.face_distance([encodeElon],encodeBillTest)

results = face_recognition.compare_faces([encodeElon],encodeElonTest)
print(results,faceDis)
results1 = face_recognition.compare_faces([encodeElon],encodeBillTest)
print(results1,faceDis1)

# cv2.imshow('Elon Musk' , imgElon)
# cv2.waitKey(0)
# cv2.imshow('Elon Musk' , testElon)
# cv2.waitKey(0)

