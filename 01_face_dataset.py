
import cv2
import os
import os.path


cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id

#if os.path.exists('usuarios/usuarios.txt'):
#    arq = open('usuarios/usuarios.txt', 'r')
#    names = arq.read().splitlines()
#    face_id = len(names)
#else:
#    face_id = 0

face_id = input('\n Digite o Id ==> ')

nome = input('\n Digite o nome do usuario e tecle ENTER ==> ')

file1 = open("usuarios/usuarios.txt","a")
file1.write(nome)
file1.write("\n")
file1.close()

print("\n [INFO] Inicializando a captura de face. Olhe para a camera e espere ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 100: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n Captura finalizada!")
cam.release()
cv2.destroyAllWindows()


