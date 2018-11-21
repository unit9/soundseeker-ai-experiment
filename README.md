# Soundseeker - AI Experiment by [UNIT9](https://www.unit9.com/)

## Overview

This is an AI experiment originally released at https://www.royalcaribbean.com/soundseeker.

The user could upload 3 holiday images and get a bespoke holiday video generated for them, including
a sound track composed to match their holiday visuals.

This has been achieved by developing a custom AI with Google TensorFlow, that was trained to
corelate visual features (patterns, colours etc.) and semantic image contents (types of objects,
people, locations) to musical features (BPM, key and more). This has been implement using three AI
networks:

1) A convolutional neural network looking at the image as a whole and learning the mapping to music.
2) A deep neural network whose input was features extracted with Google Cloud Vision and represented
as word embeddings to understand the semantic meaning, learning corelations between meaning of
items in the image and music
3) A merger neural network weighing the output of the two other networks and draing a final
conclusion.

Training has been performed with tastemakers and crowdsourcing by mapping stock images to sample
music.

In this experiment, you're given the core of the experience - the pre-trained AI, its source code
and helper scripts.

To see a full case study, visit https://www.unit9.com/project/royal-caribbean-sound-seeker.

## Setup

1) Instal python [anaconda](https://www.anaconda.com/)

2) Create environment:

  Run `conda env create --name soundseeker -f=environment.yml`

3) Activate new environment

  Windows: `activate soundseeker`

  Linux/OSX: `source activate soundseeker`

4) Setup Google Cloud Vision Key

  Windows: `set CV_KEY=YOUR_KEY`
  Linux: `export CV_KEY=YOUR_KEY`

5) Run it

`python main.py --img-path path_to_image.jpg`

6) Explore the code
