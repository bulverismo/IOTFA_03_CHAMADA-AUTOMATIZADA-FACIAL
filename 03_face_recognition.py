# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os 
from datetime import date
from datetime import datetime
import os.path

#recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer = cv2.face.createLBPHFaceRecognizer()
#recognizer.read('trainer/trainer.yml')
recognizer.load('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

arq = open('usuarios/usuarios.txt', 'r')
names = arq.read().splitlines()

cam = cv2.VideoCapture(0)
cam.set(3, 800) # set video widht
cam.set(4, 600) # set video height

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

ultimoSegundoRegistrado = -1

while True:
    
    data_e_hora_atuais = datetime.now()
    horaAtual=data_e_hora_atuais.strftime('%H%M')

    horaAtual = int(horaAtual)
# converteu para inteiro a hora atual e usou abaixo
# se horario de entrar entao da verdadeiro
    entraManha = ( (horaAtual >= 845) and (horaAtual <= 915) )
    saiMeioDia = ( (horaAtual >= 1135) and (horaAtual <= 1205) )
    entraNoite = ( (horaAtual >= 1855) and (horaAtual <= 1925) )
    saiNoite = ( (horaAtual >= 2145) and (horaAtual <= 2215) )

# se é qualquer um dos horarios de entrada então da verdadeiro
    ehHorarioDeEntrada = ( entraManha or saiMeioDia or entraNoite or saiNoite  )

#    print( entraManha,saiMeioDia,entraNoite,saiNoite  )
#    print ("eh horario de entrada agora? => ",ehHorarioDeEntrada)

#   if ( ehHorarioDeEntrada ):
    if ( 1 ):
        ret, img =cam.read()
        img = cv2.flip(img, 1) # Flip vertically
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                diretorio = "relatorio/"+str(id)
                caminho = str(diretorio)+"/"+str(id)+".txt"
                if os.path.exists(caminho):
                    arq = open(caminho, 'r')
                    texto = arq.readlines()
                    for linhaLidaAtualmente in texto :
                        ultimaLinhaLida=linhaLidaAtualmente
                    arq.close()
                    ultimaHoraRegistrada=ultimaLinhaLida[11:16]
                    data_e_hora_atuais = datetime.now()
                    horaAtual=data_e_hora_atuais.strftime('%H:%M')
                    segundoAtual = data_e_hora_atuais.strftime('%S')
                    segundoAtual = int(segundoAtual)
                    if (segundoAtual%6==0 and ultimoSegundoRegistrado != segundoAtual and segundoAtual != 0):
                        ultimoSegundoRegistrado = segundoAtual
                        print(segundoAtual)
                        data_e_hora_atuais = datetime.now()
                        data = data_e_hora_atuais.strftime('%d-%m-%Y-%H:%M')
                        cv2.imwrite(str(diretorio)+"/"+str(id)+'-'+str(data)+':'+str(segundoAtual)+".jpg", gray[y:y+h,x:x+w])
                    
                    if (ultimaHoraRegistrada != horaAtual):
                        #Registra
                        data_e_hora_atuais = datetime.now()
                        data = data_e_hora_atuais.strftime('%d-%m-%Y-%H:%M')
                        file1 = open(caminho,"a")
                        file1.write(data)
                        file1.write("\n")
                        file1.close()
                        data = str(data)
                        print (data)
                        cv2.imwrite(str(diretorio)+"/"+str(id)+'-'+str(data)+".jpg", gray[y:y+h,x:x+w])
                    
                        
                else:
                    os.mkdir(diretorio)
                    data_e_hora_atuais = datetime.now()
                    data = data_e_hora_atuais.strftime('%d-%m-%Y-%H:%M')
                    file1 = open(caminho,"a") 
                    file1.write(data)
                    file1.write("\n")
                    file1.close()
                    cv2.imwrite(str(diretorio)+"/"+str(id)+'-'+str(data)+".jpg", gray[y:y+h,x:x+w])
            else:
                id = "Desconhecido"
                confidence = "  {0}%".format(round(100 - confidence))
            
#            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
#            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    else:
        img = cv2.imread('flor.jpg',1)
        
    cv2.imshow('camera',img)
    
    

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Saindo do programa!")
cam.release()
cv2.destroyAllWindows()
