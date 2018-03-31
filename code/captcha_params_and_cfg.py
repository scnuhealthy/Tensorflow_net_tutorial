# Author: Kemo Ho
# This file is to define the paths, captcha parameters and other configuration

import os


# the size of the captcha image
WIDTH = 224
HEIGHT= 224



def get_width():
    return WIDTH

def get_height():
    return HEIGHT

model_path = './tmp'
model_tag = 'captcha4.model'
save_model = os.path.join(model_path, model_tag)           # the path to save the trained model
data_path = 'E:/test/test/data/25921_black_white'          # the data path 
