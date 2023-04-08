import kivy
kivy.require('1.0.7')

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label

class TestApp(App):

    def build(self):
        # return a Button() as a root widget
        return Button(text='hello world')
class EpicApp(App):
    # This is your "initialize" for the root wiget
    def build(self):
        # Creates that label which will just hold text.
        return Label(text="Hey there!")
class MainApp(App):
    def build(self):
        label = Label(text='Hello from Kivy',
                      size_hint=(.5, .5),
                      pos_hint={'center_x': .5, 'center_y': .5})

        return label

if __name__ == '__main__':
    # TestApp().run()
    EpicApp().run()
    # MainApp().run()