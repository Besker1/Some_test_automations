import tkinter as tk
from tkinter import filedialog
from appium import webdriver
from appium.webdriver.common.touch_action import TouchAction
from PIL import Image, ImageTk
import pytesseract
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from io import BytesIO
import time


def contrastive_loss(y_true, y_pred):
    margin = 1
    y_true = K.cast(y_true, dtype=K.floatx())
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

class ImageComparisonApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Comparison App")

        self.uploaded_image_path = None
        self.siamese_model = self.load_siamese_model()

        # Create labels
        self.label = tk.Label(master, text="Image Comparison App")
        self.label.pack()

        # Create upload button
        self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        # Create compare button
        self.compare_button = tk.Button(master, text="Compare Images", command=self.compare_images, state=tk.DISABLED)
        self.compare_button.pack()

        # Create image display
        self.image_label = tk.Label(master)
        self.image_label.pack()

    def upload_image(self):
        file_dialog = filedialog.askopenfile(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_dialog:
            self.uploaded_image_path = file_dialog.name
            self.show_uploaded_image()
            self.compare_button["state"] = tk.NORMAL

    def show_uploaded_image(self):
        if self.uploaded_image_path:
            image = Image.open(self.uploaded_image_path)
            image.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def compare_images(self):
        # Initialize Appium driver
        driver = self.launchSession()

        try:
            # Capture a screenshot
            screenshot_path = 'screenshot.png'
            self.capture_screenshot(driver, screenshot_path)

            # Perform OCR on the screenshot
            detected_text = self.perform_ocr(screenshot_path)
            print(f"Detected Text: {detected_text}")

            # Compare the screenshot with the uploaded image
            similarity = self.compare_images_with_siamese_model(screenshot_path)

            # Print the result
            if similarity:
                print("Images are the same, login must be true")
            else:
                print("failed")

        finally:
            # Close the Appium session
            driver.quit()

    def launchSession(self):
        caps = {"deviceName": "Android",
            "platformName": "Android",
            "appPackage": "com.hulu.plus",
            "appActivity": "com.hulu.features.splash.SplashActivity",
            "automationName": "UiAutomator2",
            "ensureWebviewsHavePages": True,
            'noSign': True}

        return(webdriver.Remote("http://localhost:4723/wd/hub", caps))


    time.sleep(30)


    def capture_screenshot(self, driver, screenshot_path):
        screenshot = driver.get_screenshot_as_png()
        with open(screenshot_path, 'wb') as f:
            f.write(screenshot)

    def perform_ocr(self, screenshot_path):
        screenshot = Image.open(screenshot_path)
        text = pytesseract.image_to_string(screenshot)
        return text

    def compare_images_with_siamese_model(self, screenshot_path):
        if self.uploaded_image_path is None:
            return False

        # Load and preprocess images
        uploaded_image = load_and_preprocess_image(self.uploaded_image_path, target_size=(28, 28))
        screenshot_image = load_and_preprocess_image(screenshot_path, target_size=(28, 28))

        # Assuming siamese_model is designed to accept a pair of images
        input_pair = [uploaded_image, screenshot_image]

        # Expand dimensions for batch size (None in this case)
        input_pair = [np.expand_dims(img, axis=0) for img in input_pair]

        # Concatenate the images along the batch dimension
        input_pair = [np.concatenate(input_pair, axis=0)]

        # Check if images have 3 dimensions before resizing
        input_pair = [tf.image.resize(img if img.shape[-1] == 3 else tf.squeeze(img, axis=-1), (28, 28)) for img in input_pair]

        similarity_score = self.siamese_model.predict(input_pair)

        # Set a threshold for similarity
        threshold = 0.5

        return similarity_score > threshold







    def load_siamese_model(self):
        # Load your Siamese model with custom loss
        model_path = '/Users/btelisma/Documents/Work_Org/Appium-Projects/best_model.h5'
        # Register the custom loss function in the custom object scope
        with CustomObjectScope({'contrastive_loss': contrastive_loss}):
            model = load_model(model_path)

        return model

def load_and_preprocess_image(image_path, target_size=None):
    image = keras_image.load_img(image_path, target_size=target_size)
    image_array = keras_image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparisonApp(root)
    root.mainloop()
