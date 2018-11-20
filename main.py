#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os
from ai import AiProcessor
from ai.model import MergeModel

from services.cloud_vision import CloudVisionService
from services.labels import LabelsService
import settings

def parse_args(parser: argparse.ArgumentParser):
    # Input parameters
    parser.add_argument('--img-path', type=str, required=True)    
    args = parser.parse_args()

    return args


def nice_print(result, width=55, x="#"):    
    print(x*width)
    print("{} AI OUTPUT".format(x), " "*(width-12), end='{}\n'.format(x))
    print(x*width)
    for feature, score in result.items():
        txt = "# {}: {}".format(feature, score)
        print(txt, " "*(width-len(txt)-1), end='{}\n'.format(x))        
    print(x*width)


def main():
    # Dependencies
    parser = argparse.ArgumentParser()
    cv = CloudVisionService(settings.CV_KEY)
    labels = LabelsService()
    model = MergeModel(settings.MODEL_DIR)    
    ai = AiProcessor(model, cv, labels)

    args = parse_args(parser)

    # Run model
    result = ai.process(args.img_path)

    nice_print(result)  

if __name__ == "__main__":
    main()