import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep


def browse_through_faces_folder():
    """
  this function basically goes through all the faces in the "faces folder" and then encodes them into a dictionary [the return will look like this: (name, image_encoded) ]
    """
    list_of_faces = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                faces_in_folder = fr.load_image_file("faces/" + f)
                encoding_folder_faces = fr.face_encodings(faces_in_folder)[0]
                list_of_faces[f.split(".")[0]] = encoding_folder_faces

    return list_of_faces


def test_image_encoded(img):
    """
    this function, on the other hand, encodes the "test.jpg or test.png"
    """
    face_test = fr.load_image_file("faces/" + img)
    encoding_test_image = fr.face_encodings(face_test)[0]

    return encoding_test_image


def match_face(im):
    faces = browse_through_faces_folder() #this line of code calls back the very first function and stores the dictionary (the return of the function) into the variable "faces"
    faces_encoded = list(faces.values()) #this line then makes the values within the dictionary into a list
    known_face_names = list(faces.keys()) #this line then makes the keys within the dictionary into a list

    img = cv2.imread(im, 1) #reads the test image

 
    #this next series of code determines if the test image matches any of the images in the faces folder

    unknown_face_loc = face_recognition.face_locations(img) #determines the different locations for our face images
    unknown_face_encodings = face_recognition.face_encodings(img, unknown_face_loc) #this line encodes our test image

    output_names = []
    for face_encoding in unknown_face_encodings:  
        equals = face_recognition.compare_faces(faces_encoded, face_encoding) #checks if any of the face_encoding in our "test image" matches any from our list of encodings in our "faces folder" (refer to line 37)
        name = "Cannot match face. Please try again." #this is the result if face does not match any faces in the folder

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if equals[best_match_index]:
            name = known_face_names[best_match_index]

        output_names.append(name)

    #this is the output if the test face matches any of the faces in the faces folder

        for (top, right, bottom, left), name in zip(unknown_face_loc, output_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 153, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 153, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    while True:

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return output_names 


print(match_face("test.jpg"))


