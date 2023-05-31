from typing import Any
import torch
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
import time
class Process():
    def __init__(self, model, base_model) -> None:
        self.model = model
        self.base_model = base_model
        json_file = open('./Face_Anti_Spoofing/finalyearproject_antispoofing_model_mobilenet.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model_anti = tf.keras.models.model_from_json(loaded_model_json)
        self.model_anti.load_weights('./Face_Anti_Spoofing/finalyearproject_antispoofing_model_97-0.969474.h5')
        self.labels = {}
        self.EmbeddingVectors = {}
    
    def getLabels(self):
        i = 0
        for dir in os.listdir("./Process/"):
            self.labels[i] = dir
            i += 1
        
    def getEmbeddingVector(self):
        path = "./EmbeddingVector/"
        for dir in os.listdir(path):
            vector = np.load(f"{path}{dir}", allow_pickle=True)
            self.EmbeddingVectors[dir.split(".")[0]] = vector
    
    def similarity(self, vector1, vector2):
        vector1 = vector1.reshape((1, -1))
        vector2 = vector2.reshape((1, -1))
        return cosine_similarity(vector1, vector2)

    def anti_spoofing(self, img):
        image = cv2.resize(img, (160, 160), interpolation=cv2.INTER_NEAREST)
        image = np.expand_dims(image, axis=0)
        preds = self.model_anti.predict(image, verbose=0)
        return preds[0][0]

    def compare(self, vector):
        self.getLabels()
        self.getEmbeddingVector()
        if len(self.labels) == 0:
            return "Unknown"
        results = []
        for i in range(len(self.labels)):
            result = self.similarity(vector, self.EmbeddingVectors[self.labels[i]])
            results.append(result[0][0])
        index = np.argmax(results)
        # print(results[index])
        if results[index] >= 0.55:
            return self.labels[index]
        else:
            return "Unknown"
        

class Create():
    def __init__(self, model, base_model) -> None:
        self.model = model
        self.base_model = base_model

    def ProcessData(self, name):
        path = "./Raw/"
        os.makedirs(name = f"./Process/{name}")
        dirs = os.listdir(path + name)
        for dir in dirs:
            image = cv2.imread(path + name + "\\" + dir)
            res = self.model(image)
            result = res.pandas().xyxy[0]
            if (len(result)) != 0: 
                record = result.loc[0]
                xmin = np.floor(record['xmin']).astype(int)
                ymin = np.ceil(record['ymin']).astype(int)
                xmax = np.floor(record['xmax']).astype(int)
                ymax = np.ceil(record['ymax']).astype(int)
                crop_img = image[ymin:ymax, xmin:xmax]
                resized_image = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(f"./Process/{name}/crop{dir}", resized_image)

    def CreateEmbeddingVector(self, name):
        path = f"./Process/{name}/"
        features = []
        for dir in os.listdir(path):
            img = cv2.imread(path + dir)
            img = np.expand_dims(img, axis=0)
            feature = self.base_model.predict(img, verbose = 0)
            features.append(feature)
        vector = np.mean(features, axis = 0)
        np.save(f'./EmbeddingVector/{name}.npy', vector)
