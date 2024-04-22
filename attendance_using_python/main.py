import cv2
import cvzone
import os
import pickle
import face_recognition
import numpy as np

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import numpy as np
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://face-recognition-attenda-ef4d8-default-rtdb.firebaseio.com/",
    'storageBucket': "face-recognition-attenda-ef4d8.appspot.com"
})

bucket = storage.bucket()
video=cv2.VideoCapture(0)
video.set(3,640)
video.set(4,480)

bg=cv2.imread("C:/codes/attendance_using_python/Resources/background.png")

#importing mode images into list
foldermodepath="Resources/Modes"
# print(foldermodepath)
modepathlist=os.listdir(foldermodepath)
# print(modepathlist)
imgmodelist=[]
for path in modepathlist:
        imgmodelist.append(cv2.imread(os.path.join(foldermodepath,path)))
print(len(imgmodelist))


#loading pickle encoding file
print("loading encoding file")
file=open("Encodedata.p","rb")
encodelistknownwithids=pickle.load(file)
file.close()
encodelistknown,studentId=encodelistknownwithids
# print(encodelistknown)
# print(studentId)
print("Encode file loaded")

modetype=0

counter=0
id=-1
imgStudent = []


while True:
    ret,img=video.read()

    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facecurframe=face_recognition.face_locations(imgs)
    encodecurframe=face_recognition.face_encodings(imgs,facecurframe)

    bg[162:162+480,55:55+640]=img
    bg[44:44+633,808:808+414]=imgmodelist[modetype]

    if facecurframe:
        for encodeFace, faceLoc in zip(encodecurframe, facecurframe):
            matches = face_recognition.compare_faces(encodelistknown, encodeFace)
            faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
            # print("matches", matches)
            # print("faceDis", faceDis)

            matchIndex = np.argmin(faceDis)
            # print("Match Index", matchIndex)

            if matches[matchIndex]:
                # print("Known Face Detected")
                # print(studentIds[matchIndex])
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                bg = cvzone.cornerRect(bg, bbox, rt=0)
                id = studentId[matchIndex]
                if counter == 0:
                    cvzone.putTextRect(bg, "Loading", (275, 400))
                    cv2.imshow("Face Attendance", bg)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1

        if counter != 0:

            if counter == 1:
                # Get the Data
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)
                # # Get the Image from the storage
                # blob = bucket.get_blob(f'Images/{id}.png')
                # array = np.frombuffer(blob.download_as_string(), np.uint8)
                # imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                # Update data of attendance
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                   "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)
                if secondsElapsed > 30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType = 3
                    counter = 0
                    bg[44:44 + 633, 808:808 + 414] = imgmodelist[modeType]

            if modeType != 3:

                if 10 < counter < 20:
                    modeType = 2

                bg[44:44 + 633, 808:808 + 414] = imgmodelist[modeType]

                if counter <= 10:
                    cv2.putText(bg, str(studentInfo['total_attendance']), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(bg, str(studentInfo['major']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(bg, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(bg, str(studentInfo['standing']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(bg, str(studentInfo['year']), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(bg, str(studentInfo['starting_year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(bg, str(studentInfo['name']), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    # bg[175:175 + 216, 909:909 + 216] = imgStudent

                counter += 1

                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []
                    bg[44:44 + 633, 808:808 + 414] = imgmodelist[modeType]
    else:
        modeType = 0
        counter = 0
    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", bg)
    k=cv2.waitKey(1)
    if k==ord("a"):
        break
video.release()
cv2.destroyAllWindows()


