import cv2
import face_recognition
import pickle
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://face-recognition-attenda-ef4d8-default-rtdb.firebaseio.com/",
    'storageBucket': "face-recognition-attenda-ef4d8.appspot.com"
})

#importing mode images into list
folderpath="Images"
# print(folderpath)
pathlist=os.listdir(folderpath)
# print(pathlist)
imglist=[]
studentId=[]
for path in pathlist:
        imglist.append(cv2.imread(os.path.join(folderpath,path)))
        # print(path)
        # print(os.path.splitext(path)[0])
        studentId.append(os.path.splitext(path)[0])

     #for importing images into firebase database
        fileName = f'{folderpath}/{path}'
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.upload_from_filename(fileName)
    #--------------------------------------------------------
print(studentId)
print(len(imglist))
        
        
def findencoding(imglist):
        encodelist=[]
        for img in imglist:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            encode=face_recognition.face_encodings(img)[0]
            encodelist.append(encode)

        return encodelist

print("Encoding started........")
encodelistknown=findencoding(imglist)
encodelistknownwithids=[encodelistknown,studentId]
print("Encoding Complete")


file=open("Encodedata.p","wb")
pickle.dump(encodelistknownwithids,file)
file.close()
print("File saved")
