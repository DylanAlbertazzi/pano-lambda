from __future__ import division
from __future__ import print_function

import panoramasdk
import cv2
from PIL import Image
import numpy as np
from mtcnn_cv2 import MTCNN
from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
# from keras.models import load_model

# Global Variables

HEIGHT = 160
WIDTH = 160

class people_counter(panoramasdk.base):

    def interface(self):
        return {
            "parameters":
                (
                    ("float", "threshold", "Detection threshold", 0.10),
                    ("model", "people_counter", "Model for facial recognition", "facenet_keras"),
                    ("int", "batch_size", "Model batch size", 1),
                    ("float", "person_index", "person index based on dataset used", 14),
                ),
            "inputs":
                (
                    ("media[]", "video_in", "Camera input stream"),
                ),
            "outputs":
                (
                    ("media[video_in]", "video_out", "Camera output stream"),
                )
        }


    def init(self, parameters, inputs, outputs):
        try:
            # Detection probability threshold.
            self.threshold = parameters.threshold
            # Frame Number Initialization
            self.frame_num = 0
            # Number of People
            self.number_people = 0
            # Bounding Box Colors
            self.colours = np.random.rand(32, 3)
            # Person Index for Model from parameters
            self.person_index = parameters.person_index
            # Set threshold for model from parameters 
            self.threshold = parameters.threshold

            ###Elasticsearch variables
            self.host = 'https://search-panorama-deepface-eg2tehvo2q5ssfu7gs4qb32bdu.us-west-2.es.amazonaws.com'
            self.region = 'us-west-2'
            self.service = 'es'
            self.credentials = boto3.Session().get_credentials()
            self.awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
            #Make connection to Elasticsearch
            es = Elasticsearch(
                hosts = ['https://search-panorama-deepface-eg2tehvo2q5ssfu7gs4qb32bdu.us-west-2.es.amazonaws.com:443'],
                http_auth = awsauth,
                use_ssl = True,
                verify_certs = True,
                connection_class = RequestsHttpConnection
            )

            # Load model from the specified directory.
            print("loading the model...")
            self.model = panoramasdk.model()
            self.model.open(parameters.people_counter, 1)
            print("model loaded")

            # Create input and output arrays.
            class_info = self.model.get_output(0)
            prob_info = self.model.get_output(1)
            rect_info = self.model.get_output(2)

            self.class_array = np.empty(class_info.get_dims(), dtype=class_info.get_type())
            self.prob_array = np.empty(prob_info.get_dims(), dtype=prob_info.get_type())
            self.rect_array = np.empty(rect_info.get_dims(), dtype=rect_info.get_type())

            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False

    # extract a single face from a given photograph
    def extract_face(self, image, required_size=(WIDTH, HEIGHT)):
        # # load image from file
        # image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB') #!!There could be an error here if the image from the stream is not usable by PIL library
        # convert to array
        pixels = np.asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        return face_array

    #Turns am image into a searchable array of numbers that is put into elasticsearch
    def get_embedding(self, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = self.model.predict(samples)
        self.model.flush()
        return yhat[0]

    def preprocess(self, img):
        resized = cv2.resize(img, (HEIGHT, WIDTH))

        mean = [0.485, 0.456, 0.406]  # RGB
        std = [0.229, 0.224, 0.225]  # RGB

        img = resized.astype(np.float32) / 255.  # converting array of ints to floats
        img_a = img[:, :, 0]
        img_b = img[:, :, 1]
        img_c = img[:, :, 2]

        # Extracting single channels from 3 channel image
        # The above code could also be replaced with cv2.split(img) << which will return 3 numpy arrays (using opencv)

        # normalizing per channel data:
        img_a = (img_a - mean[0]) / std[0]
        img_b = (img_b - mean[1]) / std[1]
        img_c = (img_c - mean[2]) / std[2]

        # putting the 3 channels back together:
        x1 = [[[], [], []]]
        x1[0][0] = img_a
        x1[0][1] = img_b
        x1[0][2] = img_c

        # x1 = mx.nd.array(np.asarray(x1))
        x1 = np.asarray(x1)
        return x1
   

    def entry(self, inputs, outputs):
        self.frame_num += 1

        for i in range(len(inputs.video_in)):
            stream = inputs.video_in[i]
            person_image = stream.image
            
            x1 = self.extract_face(person_image)
            embedding = self.get_embedding(x1)

            index=62
            img_path='myImage.jpeg'

            doc = {"title_vector": embedding, "title_name": img_path}
            es.create("face_recognition", id=index, body=doc)

            outputs.video_out[i] = stream

        return True


def main():
    people_counter().run()


main()
