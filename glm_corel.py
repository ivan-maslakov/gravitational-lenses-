import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.draw import *
from PIL import Image
from numba import jit

def f(pixels):
    pygame.init()
    screen = pygame.display.set_mode((l, hi))
    screen.fill((0, 0, 0))
    clock = pygame.time.Clock()
    for i in range(im.size[0]):  # for every pixel:
        for j in range(im.size[1]):
            circle(screen, pixels[i,j], (i,j), 1)
    finished = False
    schpg = 0
    crds = []
    while not finished and schpg < 2:
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
    print(crds[0] - crds[2])
    return

def delenie(pixels):
    pygame.init()
    screen = pygame.display.set_mode((l, hi))
    screen.fill((0, 0, 0))
    clock = pygame.time.Clock()
    for i in range(im.size[0]):  # for every pixel:
        for j in range(im.size[1]):
            circle(screen, pixels[i,j], (i,j), 1)
    finished = False
    schpg = 0
    crds = []
    while not finished and schpg < 12:
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

def color(p):
    return p[1] + p[2] + p[0]
im = Image.open('для ся.jpg')

def point(a, color):
    for i in np.arange(a[0] - 2, a[0] + 2):
        for j in np.arange(a[1] - 2, a[1] + 2):
            if ((i - a[0]) ** 2 + (j - a[1]) ** 2) ** 0.5 <= 2:
                pixels[i, j] = color
def holes(pixels):
    #holes_coords = []
    pygame.init()
    screen = pygame.display.set_mode((l, hi))
    screen.fill((0, 0, 0))
    clock = pygame.time.Clock()
    for i in range(im.size[0]):  # for every pixel:
        for j in range(im.size[1]):
            circle(screen, pixels[i, j], (i, j), 1)
    finished = False
    schpg = 0
    crds = []
    while not finished and schpg < 18:
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


l, hi = im.size
pixels = im.load()  # create the pixel map
d = 1
#f(pixels)
#489->3000
'''
ho_co = holes(pixels)
print(ho_co)
'''
ho_co = [2, 52, 129, 81, 175, 137, 190, 133, 235, 177, 248, 160, 294, 206, 310, 209, 355, 289, 368, 257, 404, 232, 491, 145, 532, 161, 547, 157, 591, 186, 609, 184, 651, 23, 702, 20]

#ho_co = [402, 250, 491, 147, 589, 93, 609, 85, 294, 208, 311, 207]
hole = set()
for i in range(0, len(ho_co), 2):
    if i % 4 == 0:
        for y in range(im.size[1]):
            for x in range(ho_co[i], ho_co[i+2]):
                hole.add((x, y))
'''
for t in hole:
    pixels[t[0], t[1]] = (255, 0,0)
im.show()
'''
for i in range(d + 1, im.size[0] - d - 1):  # for every pixel:
    for j in range(d + 1, im.size[1] - d - 1):
        for p in np.arange(-d, d, 1):
            for pp in np.arange(-d, d, 1):
                if pixels[i + p, j + pp][0] + pixels[i, j][1] + pixels[i, j][2] > 20:
                    pixels[i, j] = (0, 255, 0)

for x in range(5, im.size[0] - 5):
    y = im.size[1] - 5
    sch = 0
    while(y > 5 and sch < 2):
        if color(pixels[x, y]) < 20:
            point([x, y], (255, 0, 0))
            sch += 1
            y -= 40
        else:
            y -= 1
for i in range(d + 1, im.size[0] - d - 1):  # for every pixel:
    for j in range(d + 1, im.size[1] - d - 1):

        if pixels[i,j][0] < 100:
            pixels[i, j] = (255,255,255)
        else:
            pixels[i, j] = (0, 0, 0)
        if j > im.size[1] - 50:
            pixels[i, j] = (255, 255, 255)
        if i < 50:
            pixels[i, j] = (255, 255, 255)
g0 = []
for i in range(d + 1, im.size[0] - d - 1):  # for every pixel:
    for j in range(d + 1, im.size[1] - d - 1):
        if pixels[i, j][0] < 200:
            g0.append([i, j])

'''
coord = delenie(pixels)
coords = []
coords.append(0)
coords.append(coord[1])
for i in range(len(coord)):
    coords.append(coord[i])
coords.append(im.size[0])
coords.append(coord[len(coord) - 1])
print(coords)
'''
#coords = [0, 203, 36, 203, 151, 227, 243, 284, 321, 345, 355, 368, 423, 329, 503, 213, 571, 177, 607, 137, 647, 79, 705, 79]
coords = [0, 192, 24, 192, 194, 243, 288, 324, 324, 365, 403, 359, 483, 240, 543, 207, 577, 174, 605, 153, 615, 125, 646, 83, 657, 72, 705, 72]

tochky = []
g1 = []
g2 = []
for i in range(0, len(coords) - 2, 2):
    tochky.append([coords[i], coords[i+1]])
print(tochky)
for i in range(len(tochky) - 1):
    k = (tochky[i][1] - tochky[i + 1][1]) / (tochky[i][0] - tochky[i + 1][0])
    for x in range(tochky[i][0], tochky[i+1][0]):
        for t in g0:
            if t[0] == x and t[1] > tochky[i][1] + k * (x - tochky[i][0]):
                g2.append(t)
            if t[0] == x and t[1] < tochky[i][1] + k * (x - tochky[i][0]):
                g1.append(t)
g1set = set()
g2set = set()
for t in g1:
    pixels[t[0], t[1]] = (255, 0, 0)
for t in g2:
    pixels[t[0], t[1]] = (0, 255, 0)
for t in hole:
    pixels[t[0], t[1]] = (0, 0, 255)
im.show()
for t in g1:
    pixels[t[0], t[1]] = (255, 255, 255)
for t in hole:
    pixels[t[0], t[1]] = (0, 0, 255)
sdvx = -25
#sdvx = -12
sdvy = 120
hole_ = set()
for t in hole:
    hole_.add((t[0] + sdvx, t[1]))
for t in g1:
    t[0] += sdvx
    t[1] += sdvy
for t in g1:
    pixels[t[0], t[1]] = (255, 0, 0)
for t in hole:
    pixels[t[0], t[1]] = (255, 255, 255)
for t in hole_:
    pixels[t[0], t[1]] = (255, 255, 255)

im.show()