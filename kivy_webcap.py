import os
import cv2
import numpy as np
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.core.window import Window
import tensorflow as tf
import time

with open('./cv_models/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (640, 480)
        play: False
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
''')


class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png(f"./Captured/img_{timestr}.png")
        print("Captured")


class CamApp(App):
    def build(self):
        self.web_cam = Image(size_hint=(1,.8))
        self.button_check = Button(
            text="Verify", 
            on_press=self.verify, 
            size_hint=(0.1,0.1))
        self.button_close = Button(
            text="Exit", 
            on_press=self.close_application, 
            size_hint=(0.1,0.1))
        self.verification_label = Label(
            text="Verification Uninitiated", 
            size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button_check)
        layout.add_widget(self.button_close)
        layout.add_widget(self.verification_label)
        self.model = cv2.dnn.readNet(model='./cv_models/frozen_inference_graph.pb',
                        config='./cv_models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout
    
    def update(self, *args):
        ret, frame = self.capture.read()
        # frame = frame[0:120+250, 200:200+250, :]
        frame = frame[0:, 0:, :]
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
        
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100,100))
        img = img / 255.0
        return img
    
    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.5
        verification_threshold = 0.5

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'IMG_20230407_073958.png')
        
        # while self.capture.isOpened():
        ret, frame = self.capture.read()
        image = frame
        image_height, image_width, _ = image.shape
        blob = cv2.dnn.blobFromImage(
                image=image, 
                size=(300, 300), 
                mean=(104, 117, 123), 
                swapRB=True)
        self.model.setInput(blob)
        output = self.model.forward()
        for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > .4:
                    # get the class id
                    class_id = detection[1]
                    # map the class id to the class 
                    class_name = class_names[int(class_id)-1]
                    color = COLORS[int(class_id)]
                    # get the bounding box coordinates
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    # get the bounding box width and height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    # draw a rectangle around each detected object
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                    # put the class name text on the detected object
                    cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    # put the FPS text on top of the frame
                    # cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            # results.append(result)
            # cv2.imshow('image', image)
            # out.write(image)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
        cv2.imwrite("./application_data/verification_images/img.jpg", image)
        
        # else:
        #     break
        # Detection Threshold: Metric above which a prediciton is considered positive 
        # detection = np.sum(np.array(results) > detection_threshold)

        # Verification Threshold: Proportion of positive predictions / total positive samples 
        # verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        # verified = verification > verification_threshold
        # self.verification_label.text = 'Verified' if verified else 'Unverified'

        # Log out details
        try:
            Logger.info(f"Class Name: {class_name}")
            Logger.info(f"Confidence: {confidence}")
            # Logger.info(f"Verification: {verification}")
            # Logger.info(f"Verified: {verified}")
            return image, class_name
            # return image
        except Exception as e:
            print(f"[ERROR] {e}")
            
    
    def close_application(self, *args):
        # App.get_running_app().stop(self)
        App.stop(self)
        Window.close()
        # return CameraClick()

if __name__ == '__main__':
    with open('./cv_models/object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')
    CamApp().run()