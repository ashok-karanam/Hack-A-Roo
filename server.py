from flask import Flask,render_template,Response
import cv2
import numpy as np
import winsound
import tensorflow as tf

frequency = 2500 # Set Frequency
duration = 1000

img_size = 224
new_model = tf.keras.models.load_model("my_model2.h5")

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():

    counter = 0
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            detector=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            eyes = eye_cascade.detectMultiScale(gray,1.1,4)
            for x,y,w,h in eyes:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
                eyess = eye_cascade.detectMultiScale(roi_gray)
                if len(eyess) == 0:
                    print("eyes are not detected")
                else:
                    for (ex,ey,ew,eh) in eyess:
                        eyes_roi = roi_color[ey: ey+eh, ex:ex+ew]
                
            final_image = cv2.resize(eyes_roi, (224,224))
            final_image = np.expand_dims(final_image, axis = 0)
            final_image = final_image/255.0
            font = cv2.FONT_HERSHEY_SIMPLEX
    
            Predictions = new_model.predict(final_image)
            if (Predictions>0):
                status = "Open Eyes"
                cv2.putText(frame,status,
                    (150,150),
                   font, 3, 
                   (0,255,0),
                   2,cv2.LINE_4)
                x1,y1,w1,h1 = 0,0,175,75  # Draw black rectangle
                cv2.rectangle(frame, (x1,x1), (x1+w1,y1+h1),(0,0,0), -1)
                cv2.putText(frame, 'Active',(x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        
            else:
                counter = counter + 1
                status = "Closed Eyes"
                cv2.putText(frame,status,
                    (150,150),
                   font, 3, 
                   (0,255,0),
                   2,cv2.LINE_4)
                x1,y1,w1,h1 = 0,0,175,75  # Draw black rectangle
                cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (0,0,255), 2)
    
            if counter>5:
                x1,y1,w1,h1 = 0,0,175,75
                cv2.rectangle(frame,(x1, x1), (x1+w1,y1+h1), (0,0,0), -1) #Black Rectangle
                cv2.putText(frame, 'Sleep Alert !!',(x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                winsound.Beep(frequency,duration)
                counter = 0

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)