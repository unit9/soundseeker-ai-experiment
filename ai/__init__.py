#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tools import load_image, get_image_stats
from services.interfaces import ICloudVision, ILabels

class AiProcessor(object):


    def __init__(self, model, cv: ICloudVision, postprocessor: ILabels=None):        
        self.model = model
        self.cv = cv
        self.postprocessor = postprocessor
        print(self.postprocessor)


    def process(self, img_path: str):
        img = load_image(img_path)

        # Google Vision API 
        features = self.cv.recognize_features(img)                 
        #calculate image statistics
        stats = get_image_stats(img)

        #run tf model
        predictions = self.model.run(img, features, stats)

        #postprocess the predictions
        if self.postprocessor:
            predictions = self.postprocessor.adjust_output(predictions)

        return predictions
