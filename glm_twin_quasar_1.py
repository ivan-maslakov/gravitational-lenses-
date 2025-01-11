import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.draw import *
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from numba import jit

im = Image.open('heic1702g.jpg')
pr_color = 90
rmk = 1
r = 1
g = 1
b = 1
horizontal = True
#horizontal = False

def conv1(x, y):
    #из нормальных в пиллоу
    x = x + l // 2
    y = hi // 2 - y
    return (x, y)

def conv2(x, y):
    x = x - l // 2
    y = hi // 2 - y
    return (x, y)


l, hi = im.size
pixels = im.load() # create the pixel map
pix0 = []

pix = []
for i in range(im.size[0]): # for every pixel:
    for j in range(im.size[1]):
        pix.append([np.array([i, j]), pixels[i, j]])
pix0 = pix
'''
for i in range(im.size[0]): # for every pixel:
    for j in range(im.size[1]):
        #pixels[i, j] = ((255 - pixels[i, j][0]) // 2, (255 - pixels[i, j][1]) // 2, (255 - pixels[i, j][2]) // 2)
        #rmk = 1/4
        prr_color = pr_color * (r + g + b) / 3
        #obr1 - 200
        if (2 * pixels[i, j][0] + 1 * pixels[i, j][1] + 1 * pixels[i, j][2]) > prr_color or (abs(i - im.size[0] / 2) / im.size[0] > rmk or abs(j - im.size[1] / 2) / im.size[1] > rmk):
            pixels[i, j] = (255, 255 ,255)
        #elif pixels[i, j][2] < 0:
            #pixels[i, j] = (0, 0, 0)
        else:
            x, y = conv2(i, j)
            pix.append([np.array([i, j]), pixels[i, j]])
pix0 = pix
'''
def take_center_zasvetky():
    pygame.init()
    screen = pygame.display.set_mode((l, hi))
    screen.fill((0, 0, 0))
    clock = pygame.time.Clock()
    for p in pix:
        circle(screen, p[1], (p[0][0], p[0][1]), 1)
    finished = False
    schpg = 0
    crds = []
    while not finished and schpg < 3:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finished = True

                pygame.display.update()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                crds.append(event.pos[0])
                crds.append(event.pos[1])

                schpg += 1
        pygame.display.update()
    clock.tick(2)
    pygame.quit()
    return crds


def take_profile(center, horizontal):
    if not horizontal:
        arx = []
        ary = []
        x = center[0]
        for i in range(min(im.size[0], im.size[1])):
            if 0 < x < im.size[0]:
                arx.append(i)
                #ary.append((r * pixels[i, x][0], g * pixels[i, x][1], b * pixels[i, x][2]))
                ary.append((r * pixels[x, i][0], g * pixels[x, i][1], b * pixels[x, i][2]))
        return [arx, ary]
    arx = []
    ary = []
    x = center[0]
    for i in range(min(im.size[0], im.size[1])):
        if 0 < x < im.size[0]:
            arx.append(i)
            # ary.append((r * pixels[i, x][0], g * pixels[i, x][1], b * pixels[i, x][2]))
            ary.append((r * pixels[i, x][0], g * pixels[i, x][1], b * pixels[i, x][2]))
    return [arx, ary]


def without_profile(profile, center):
    for i in range(l):
        for j in range(hi):
            x = i - center[0]
            y = j - center[1]
            r_pr = len(profile[0]) // 2
            if ((x) ** 2 + (y) ** 2) < r_pr ** 2:
                color = (pixels[i, j][0] - profile[1][r_pr - int((x ** 2 + y ** 2) ** 0.5)][0], pixels[i, j][1] - profile[1][r_pr - int((x ** 2 + y ** 2) ** 0.5)][1], pixels[i, j][2] - profile[1][r_pr - int((x ** 2 + y ** 2) ** 0.5)][2])
                #for k in range(2):
                    #color[k] = pixels[i, j][k] - profile[1][r_pr - int((x ** 2 + y ** 2) ** 0.5)][k]
                pixels[i, j] = color

#pix = []
'''
for i in range(im.size[0]): # for every pixel:
    for j in range(im.size[1]):
        #pixels[i, j] = ((255 - pixels[i, j][0]) // 2, (255 - pixels[i, j][1]) // 2, (255 - pixels[i, j][2]) // 2)
        #rmk = 1/4
        prr_color = pr_color * (r + g + b) / 3
        #obr1 - 200
        if (2 * pixels[i, j][0] + 1 * pixels[i, j][1] + 1 * pixels[i, j][2]) > prr_color or (abs(i - im.size[0] / 2) / im.size[0] > rmk or abs(j - im.size[1] / 2) / im.size[1] > rmk):
            pixels[i, j] = (255, 255 ,255)
        #elif pixels[i, j][2] < 0:
            #pixels[i, j] = (0, 0, 0)
        else:
            x, y = conv2(i, j)
            pix.append([np.array([i, j]), pixels[i, j]])
'''
#center = take_center_zasvetky()
#profile = take_profile(center, horizontal)
#without_profile(profile, center)
cr = take_center_zasvetky()
print(cr)

im.show()