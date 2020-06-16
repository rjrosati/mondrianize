import numpy as np
import cv2
import argparse
import os

color_library = {
        'red'    : (217, 35, 21),
        'white'  : (255,255,255),
        'blue'   : (  2, 95,162),
        'yellow' : (237,219,111)
        }
border_color = (7,26,20)
class Art:
    def __init__(self,rects,colors):
        self.rects = rects
        self.colors = colors
    def spawn(self):
        newrects = rects
        newcolors = colors
        return Art(newrects,newcolors)

def fileexists(s):
    if not os.path.isfile(s):
        raise argparse.ArgumentError("{} is not a file!".format(s))
    return s
parser = argparse.ArgumentParser(description='Take an image, and create a Piet Mondrian-esque representation of it.')
parser.add_argument('image',type=fileexists,help='the image to model')
args = parser.parse_args()

img = cv2.imread(args.image,cv2.IMREAD_COLOR)
