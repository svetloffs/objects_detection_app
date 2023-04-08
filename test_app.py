from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, NumericProperty
from kivy.lang import Builder
import cv2

Builder.load_string('''
<KivyCamera>:
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

class KivyCamera(Image):
    source = ObjectProperty()
    fps = NumericProperty(30)

    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self._capture = None
        if self.source is not None:
            self._capture = cv2.VideoCapture(self.source)
        Clock.schedule_interval(self.update, 1.0 / self.fps)

    def on_source(self, *args):
        if self._capture is not None:
            self._capture.release()
        self._capture = cv2.VideoCapture(self.source)

    @property
    def capture(self):
        return self._capture

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
            )
            image_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
            self.texture = image_texture


class CamApp(App):
    pass


if __name__ == "__main__":
    CamApp().run()