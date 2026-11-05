#Import Kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

#Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

#Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

#Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np
from face_utils import crop_face

#Build app and layout
class CamApp(App):

    def build(self):
        #Main layout components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text = "Verify", on_press = self.verify, size_hint=(1,.1))
        self.verification_label = Label(text = "Verification Uninitiated", size_hint=(1,.1))

        #Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #Load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamesemodelv3.h5', custom_objects={'L1Dist':L1Dist})

        #Setup video capture device
        self.capture = cv2.VideoCapture(0)
        #Looping the update function so that webcam continuously updates feed from our webcam
        Clock.schedule_interval(self.update, 1.0/33.0)


        return layout
    
    #Run continuously to get the webcam feed while the code is running
    def update(self, *args):
        
        #Read frame from opencv
        ret, frame = self.capture.read() 
        frame = frame[120:120+250, 200:200+250, :]

        #Flip the vertical image from webcam into horizontal format and convert the image into texture which is a format understood by the Kivy app and our GPU
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt = 'ubyte')
        self.web_cam.texture = img_texture

    #Preprocessing the image - Scaling and Resizing the image
    #Load image from file path and convert to 100x100px
    def preprocess(self, file_path):
        #Read in image from file path
        byte_img = tf.io.read_file(file_path)
        #Load in the image
        img = tf.io.decode_jpeg(byte_img)

        #Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        #Scale the image to be between 0 and 1
        img = img / 255.0
        #Returns the image
        return img

    #Bring the verification function from main.py
    def verify(self, *args):

        #Specify thresholds
        detection_threshold = 0.7
        verification_threshold = 0.3

        #Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        # ret, frame = self.capture.read()
        # frame = frame[120:120+250, 200:200+250, :]
        # cv2.imwrite(SAVE_PATH, frame)
        face = None
        for _ in range(10):  # try up to 10 frames
            ret, frame = self.capture.read()
            if ret:
                face = crop_face(frame)
                if face is not None:
                    break
                print("Face found:", face is not None)
        if face is None:
            self.verification_label.text = 'No face detected'
            return None, False
        cv2.imwrite(SAVE_PATH, face)

        ver_dir = os.path.join('application_data', 'verification_images')
        image_names = os.listdir(ver_dir)

        input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_imgs = [self.preprocess(os.path.join(ver_dir, name)) for name in image_names]

        input_batch = np.stack([input_img] * len(validation_imgs))
        validation_batch = np.stack(validation_imgs)

        results = self.model.predict([input_batch, validation_batch], batch_size=16, verbose=0)

        detection = np.sum(results > detection_threshold)
        verification = detection / len(image_names)

        verified = verification > verification_threshold

        print(f"Matches: {detection} out of {len(image_names)}")
        print("Scores:", np.round(results.flatten(), 2))

        #Set verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        return results, verified

        #Log out details

        print(Logger.info(results))
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

if __name__ == "__main__":
    CamApp().run()