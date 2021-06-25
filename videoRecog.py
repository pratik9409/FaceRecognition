import cv2
import face_recognition
import os


known_faces_fol = 'known_faces'
unknown_faces_fol = 'unknown_faces'



tolerance = 0.6

model = 'hog' #if have GPU use "cnn" or if want to run only on cpu use "hog" 

video = cv2.VideoCapture(0)
#video = cv2.VideoCapture()
#address = "https://192.168.0.101:8080/video"
#video.open(address)



known_faces = []
known_names = []

for name in os.listdir(known_faces_fol):
    for filename in os.listdir(f"{known_faces_fol}/{name}"):
        img = face_recognition.load_image_file(f"{known_faces_fol}/{name}/{filename}")
        encoding = face_recognition.face_encodings(img)[0]
        known_faces.append(encoding)
        known_names.append(name)
       
#for filename in os.listdir(unknown_faces_fol):
while True:
    #print(filename)
    #img = face_recognition.load_image_file(f"{unknown_faces_fol}/{filename}")
    ret, img = video.read()
    locations = face_recognition.face_locations(img, model=model)
    encodings = face_recognition.face_encodings(img, locations)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match Found: {match}")
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            cv2.rectangle(img, top_left, bottom_right,(0,255,0), 2)
            
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(img, top_left, bottom_right,(0,255,0), 2)
            cv2.putText(img, match, (face_location[3]+10, face_location[0]+90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0), 2)
    
    cv2.imshow(filename, img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
            

