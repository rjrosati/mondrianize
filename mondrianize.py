import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
import random

color_library = {
        'red'    : (217, 35, 21),
        'white'  : (255,255,255),
        'blue'   : (  2, 95,162),
        'yellow' : (237,219,111),
        'black'  : (  0,  0,  0),
        }
border_color = (7,26,20)
border_thickness = 3

class Art:
    def __init__(self,model,rects=None,colors=None):
        self.model = model
        if rects is None:
            self.rects = self._root_rect()
            #self.rects = self._random_rects()
        else:
            self.rects = rects
        if colors is None:
            self.colors = []
            for _ in self.rects:
                self.colors.append(random.choice(list(color_library.keys())))
        else:
            self.colors = colors
    def _splitx(self,rect):
        ul,lr = rect
        a = random.randint(ul[0]+1,lr[0]-1)
        return [ [ul, (a,lr[1])] , [ (a,ul[1]), lr] ]
    def _splity(self,rect):
        ul,lr = rect
        a = random.randint(ul[1]+1,lr[1]-1)
        return [ [ul, (lr[0],a) ] , [ (ul[0],a), lr] ]
    def _root_rect(self):
        l,w,_ = self.model.shape
        return [ [(0,0),(w,l)] ]
    def _random_rects(self):
        l,w,_ = self.model.shape
        rects = [ [(0,0),(w,l)] ]
        for _ in range(6):
            irects = []
            for r in rects:
                lenx = r[1][0] - r[0][0]
                leny = r[1][1] - r[0][1]
                wx = abs(lenx) if abs(lenx) > 1 else 0
                wy = abs(leny) if abs(leny) > 1 else 0
                wi = max(w,l)/10
                wx,wy,wi = np.array([wx,wy,wi])/(wx+wy+wi)
                func = random.choices(
                        population=[self._splitx,self._splity,lambda x: [x]],
                        weights = [wx,wy,wi],
                        k = 1)[0]
                irects.extend(func(r))
            rects = irects
        return rects
    def spawn(self):
        r,c = random.choice(list(zip(self.rects,self.colors)))
        lenx = r[1][0] - r[0][0]
        leny = r[1][1] - r[0][1]
        l,w,_ = self.model.shape
        wx = abs(lenx) if abs(lenx) > 1 else 0
        wy = abs(leny) if abs(leny) > 1 else 0
        wi = max(w,l)/10
        wx,wy,wi = np.array([wx,wy,wi])/(wx+wy+wi)
        func = random.choices(
                population=[self._splitx,self._splity,lambda x: [x]],
                weights = [wx,wy,wi],
                k = 1)[0]
        newr = func(r)
        newc = []
        for _ in newr:
            newc.append(random.choice(list(color_library.keys())))
        newrects = self.rects[:]
        newrects.remove(r)
        newrects.extend(newr)
        newcolors = self.colors[:]
        newcolors.remove(c)
        newcolors.extend(newc)
        return Art(self.model,newrects,newcolors)
    def draw(self):
        img = np.zeros_like(self.model)
        for rect,color in zip(self.rects,self.colors):
            cv2.rectangle(img,rect[0],rect[1],color_library[color],thickness=-1)
            cv2.rectangle(img,rect[0],rect[1],border_color,thickness=border_thickness)
        return img
    def error(self):
        img = self.draw()
        rbar = (img[:,:,0] + self.model[:,:,0])/2
        dR = img[:,:,0] - self.model[:,:,0]
        dG = img[:,:,1] - self.model[:,:,1]
        dB = img[:,:,2] - self.model[:,:,2]
        chi2 = np.sum( (2 + rbar/256)*dR**2 + 4*dG**2 + (2+(255-rbar)/256)*dB**2)
        return chi2

def fileexists(s):
    if not os.path.isfile(s):
        raise argparse.ArgumentError("{} is not a file!".format(s))
    return s

parser = argparse.ArgumentParser(description='Take an image, and create a Piet Mondrian-esque representation of it.')
parser.add_argument('image',type=fileexists,help='the image to model')
args = parser.parse_args()

model = cv2.imread(args.image,cv2.IMREAD_COLOR)
model = cv2.cvtColor(model,cv2.COLOR_BGR2RGB)
surviving_pop = 10
num_generations = 500
num_children = 100
genepool = []
for i in range(int(np.ceil(surviving_pop/num_children))):
    genepool.append(Art(model))
for i in range(num_generations):
    temppool = []
    fitness = []
    for art in genepool:
        for _ in range(num_children):
            spawn = art.spawn()
            temppool.append(spawn)
            fitness.append(1/spawn.error())
    fitness = np.array(fitness)
    temppool = np.array(temppool)
    surv = np.argpartition(fitness,-surviving_pop)[-surviving_pop:]
    genepool = temppool[surv]
fitness = []
for art in genepool:
    fitness.append(1/art.error())
fitness = np.array(fitness)
result = genepool[np.argmax(fitness)].draw()
fig, ax = plt.subplots(2,1)
ax[0].imshow(model)
ax[1].imshow(result)
plt.show()


result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
cv2.imwrite(args.image+".mond.jpg",result)
