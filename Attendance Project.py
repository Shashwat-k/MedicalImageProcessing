
3# import cv2
# import numpy as np
# import face_recognition
# import os
# from datetime import datetime
# import time
#
# # from PIL import ImageGrab
#
# path = 'Attendance'
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)
# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
# images.append(curImg)
# classNames.append(os.path.splitext(cl)[0])
#
#
#
# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList
# #
# def markAttendance(name):
#     with open('Attendance.csv', 'r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             dtDate = now.strftime('%d:%m:%y')
#
#             f.writelines(f'\n{name},{dtString},{dtDate}')
#
#     #### FOR CAPTURING SCREEN RATHER THAN WEBCAM
#     # def captureScreen(bbox=(300,300,690+300,530+300)):
#     #     capScr = np.array(ImageGrab.grab(bbox))
#     #     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     #     return capScr
#
# encodeListKnown = findEncodings(images)
# print('Encoding Complete')
#
# cap = cv2.VideoCapture(0)
# t_end = time.time() + 60 * 1
# while time.time() < t_end:
#     success, img = cap.read()
#     # img = captureScreen()
#     imgS = cv2.resize(img, (0, 0), None, 0.25,0.25 )
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#
#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
#
#     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#         matchIndex = np.argmin(faceDis)
#
#
#     if matches[matchIndex]:
#         name = classNames[matchIndex].upper()
#         print(name)
#         y1, x2, y2, x1 = faceLoc
#         y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#         cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#         markAttendance(name)
#     else:
#         y1, x2, y2, x1 = faceLoc
#         y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
#         cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 0), cv2.FILLED)
#
#
#     cv2.imshow('Webcam', img)
#     cv2.waitKey(1)
#     time.sleep(5)

import face_recognition
import cv2
import numpy as np
import os
video_capture = cv2.VideoCapture(0)

path = 'Attendance'
images = []
known_face_encodings = []
known_face_names = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    curImg_encoding = face_recognition.face_encodings(curImg)[0]
images.append(curImg)
known_face_encodings.append(curImg_encoding)
known_face_names.append(os.path.splitext(cl)[0])

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/2 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/8 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()