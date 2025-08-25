import cv2

def main():
    
    #Haar Cascade method for object detection
    haarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: no webcam")
        exit()

    while True:
        
        #read from webcam
        out, image = cam.read()
        if not out:
            print("Error: couldn't read image")
            break
        
        #make image grayscale
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = haarCascade.detectMultiScale(grayed, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        cv2.imshow('Face Detection', image)
        
        #End on pressing Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()