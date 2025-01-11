import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.draw import *
from PIL import Image
#from numba import jit
im = Image.open('g0.png')

pixels = list(im.getdata())
l, hi = im.size


def mu(x):
    mu = (1 - x ** -4)
    if mu > 0:
        return mu
    else:
        return 1 / abs(mu)


def d_ang(z1, z2, H0=70, Omega_m=0.3, Omega_la=0.7):
    return 300000 / (1 + z2) * integral(h_pod_int, z1, z2)


def h_pod_int(z, H0=70, Omega_m=0.3, Omega_la=0.7):
    return 1 / h(z)


def h(z, H0=70, Omega_m=0.3, Omega_la=0.7):
    return H0 * (Omega_m * (1 + z) ** 3 + Omega_la) ** 0.5


def integral(f, x1, x2):
    tochn = 10000
    if x1 != 0:
        step = min(x1 / tochn, (x2 - x1) / tochn)
    else:
        step = (x2 - x1) / tochn
    ans = 0
    for i in range(tochn):
        ans += step * f(x1 + i * step)
    return ans


def image_curve(pixels, l, h, xc, yc, lv, hv, r0):
    for ind in range(len(pixels)):
        x_na_kart = ind % l
        y_na_kart = ind // l
        x0 = xc - lv / 2 + x_na_kart / l * lv - sc_s
        y0 = yc - hv / 2 + y_na_kart / h * hv - sc_s
        x, y = vision(ml, zl, zs, x0 / sc_s * pole_zr, y0 / sc_s * pole_zr)
        x = x / pole_zr * sc_s
        y = y / pole_zr * sc_s
        x1 = x[0]
        x2 = x[1]
        y1 = y[0]
        y2 = y[1]
        r = max((x1 ** 2 + y1 ** 2) ** 0.5 / (x0 ** 2 + y0 ** 2) ** 0.5 * r0,
                (x2 ** 2 + y2 ** 2) ** 0.5 / (x0 ** 2 + y0 ** 2) ** 0.5 * r0, 1)
        r = min(1, r)
        rr = max((x1 ** 2 + y1 ** 2) ** 0.5 / (x0 ** 2 + y0 ** 2) ** 0.5 * r0,
                 (x2 ** 2 + y2 ** 2) ** 0.5 / (x0 ** 2 + y0 ** 2) ** 0.5 * r0)
        rr = min(1, r)
        x1 += sc_s
        x2 += sc_s
        y1 += sc_s
        y2 += sc_s
        x0 += sc_s
        y0 += sc_s
        polygon(screen, pixels[ind], [(x1 - r, y1 + r), (x1 - r, y1 - r), (x1 + r, y1 - r), (x1 + r, y1 + r)])
        polygon(screen, pixels[ind], [(x2 - r, y2 + r), (x2 - r, y2 - r), (x2 + r, y2 - r), (x2 + r, y2 + r)])
        # circle(screen, pixels[ind], (x1, y1), r)
        # circle(screen, pixels[ind], (x2, y2), r)
        # circle(screen, pixels[ind], (x0, y0), 5)
        pygame.display.update()
        if ind % 1000 == 0:
            print(x1, y1, x2, y2)
            print(x0, y0)
            print(ind, 10000000000000000000000000000000000000000)


H0 = 70
Omega_m = 0.3
Omega_la = 0.7
c = 300000000

sc_s = 400

step = 10 ** -7  # радиан в клеточке
x0 = 0
y0 = 3.4
ml = 10 ** 7 * 2 * 10 ** 30
ml = 10 ** 12 * 2 * 10 ** 30
G = 6.67 * 10 ** -11
pole_zr = 0.000025
l_im = pole_zr / 2
im_pix = sc_s * l_im / pole_zr / l
h_im = hi * l_im / l
l_im_vis = l_im / pole_zr * sc_s
h_im_vis = h_im / pole_zr * sc_s
zl = 0.000001
zl = 0.5
zs = 2 * 0.000001
zs = 1
r0 = min(l_im_vis / l, h_im_vis / hi)
ds = d_ang(0, zs)
dl = d_ang(0, zl)
dls = d_ang(zl, zs)

vsp1 = 150000000000 * 1000000 * 206265
teta_e = (4 * G * ml / c ** 2 * dls / dl / ds / vsp1) ** 0.5
revis = teta_e / pole_zr * sc_s
ds = d_ang(0, zs)
dl = d_ang(0, zl)
dls = d_ang(zl, zs)

teta_e = (4 * G * ml / c ** 2 * dls / dl / ds / vsp1) ** 0.5


def vision(x0, y0):
    b = (x0 ** 2 + y0 ** 2) ** 0.5 / teta_e

    r_1 = (b + (b ** 2 + 4) ** 0.5) / 2 * teta_e
    r_2 = (b - (b ** 2 + 4) ** 0.5) / 2 * teta_e

    if x0 == 0 and y0 == 0:
        sin = 1
        cos = 1
    else:
        sin = y0 / (x0 ** 2 + y0 ** 2) ** 0.5
        cos = x0 / (x0 ** 2 + y0 ** 2) ** 0.5
    return [r_1 * cos, r_1 * sin, mu(r_1 / teta_e)], [r_2 * cos, r_2 * sin, mu(r_2 / teta_e)]


# pygame.init()

FPS = 5
# screen = pygame.display.set_mode((2 * sc_s, 2 * sc_s))

# screen.fill((255, 255, 255))

# circle(screen, (0, 0, 255), (sc_s, sc_s), 10)

# pygame.display.update()
# clock = pygame.time.Clock()
finished = False
x1 = sc_s
y1 = sc_s
x2 = sc_s
y2 = sc_s
# screen.fill((0, 0, 0))
vsp = 0
'''
while not finished:
    circle(screen, (255, 0, 0), (sc_s, sc_s), 5)
    circle(screen, (0, 255, 255), (x0, y0), 0)
    s = 'сторона квадрата = ' + str(2 * pole_zr) + 'Rad'
    f1 = pygame.font.Font(None, 30)
    text1 = f1.render(s, True, (255, 0, 0))
    screen.blit(text1, (15, 50))

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            finished = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x0 = event.pos[0]
            y0 = event.pos[1]
            circle(screen, (0, 255, 255), (x0, y0), 3)
            image_curve(pixels, l, hi, x0, y0, l_im_vis, h_im_vis, r0)
            print(l, hi, x0, y0, l_im_vis, h_im_vis)
            pygame.display.update()
'''


def image_0(pixels, l, h, xc, yc, lv, hv, lines, par):
    for ind in range(len(pixels)):
        # print('has been drawn ', int(ind / len(pixels) * 100), '%')
        line = lines[int(xc - lv / 2 + ind % l / l * lv) * (2 * sc_s + 1) + int(yc - hv / 2 + ind // l / h * hv)]
        x1, y1, mu1, x2, y2, mu2 = line
        # im_pix = 1

        r1 = im_pix * mu1 ** 0.5 * par
        r2 = im_pix * mu2 ** 0.5 * par

        # c1 = (255 - mu1 / 500 * (255 - pixels[ind][0]), 255 - mu1 / 500 * (255 - pixels[ind][1]), 255 - mu1 / 500 * (255 - pixels[ind][2]))

        # c2 = (255 - mu2 / 500 * (255 - pixels[ind][0]), 255 - mu2 / 500 * (255 - pixels[ind][1]), 255 - mu2 / 500 * (255 - pixels[ind][2]))
        # w = 0.0
        # p = 10
        mu1 = (np.log(mu1) + 6.9) / 11
        mu2 = (np.log(mu2) + 6.9) / 11
        # mu1 = np.log(mu1)
        # mu2 = np.log(mu2)
        c1 = (mu1 * pixels[ind][0], mu1 * pixels[ind][1], mu1 * pixels[ind][2])
        c2 = (mu2 * pixels[ind][0], mu2 * pixels[ind][1], mu2 * pixels[ind][2])
        # c1, c2 = pixels[ind], pixels[ind]
        '''
        if r1 > 10 or r1 < 1:
            circle(screen, c1, (x1, y1), 3)
        if r2 > 10 or r2 < 1:
            circle(screen, c2, (x2, y2), 3)
        else:
            w = 1
        '''

        polygon(screen, c1, [(x1 - r1, y1 + r1), (x1 - r1, y1 - r1), (x1 + r1, y1 - r1), (x1 + r1, y1 + r1)])
        polygon(screen, c2, [(x2 - r2, y2 + r2), (x2 - r2, y2 - r2), (x2 + r2, y2 - r2), (x2 + r2, y2 + r2)])
        # circle(screen, pixels[ind], (x1, y1), r)
        # circle(screen, pixels[ind], (x2, y2), r)
        # circle(screen, pixels[ind], (x0, y0), 5)


def image_1(pixels, l, h, xc, yc, lv, hv, lines, par):
    for ind in range(len(pixels)):
        # print('has been drawn ', int(ind / len(pixels) * 100), '%')
        line = lines[int(xc - lv / 2 + ind % l / l * lv) * (2 * sc_s + 1) + int(yc - hv / 2 + ind // l / h * hv)]
        x1, y1, mu1, x2, y2, mu2 = line
        # im_pix = 1

        r1 = im_pix * par
        r2 = im_pix * par

        # c1 = (255 - mu1 / 500 * (255 - pixels[ind][0]), 255 - mu1 / 500 * (255 - pixels[ind][1]), 255 - mu1 / 500 * (255 - pixels[ind][2]))

        # c2 = (255 - mu2 / 500 * (255 - pixels[ind][0]), 255 - mu2 / 500 * (255 - pixels[ind][1]), 255 - mu2 / 500 * (255 - pixels[ind][2]))
        # w = 0.0
        # p = 10
        mu1 = (np.log(mu1) + 6.9) / 11
        mu2 = (np.log(mu2) + 6.9) / 11
        # mu1 = np.log(mu1)
        # mu2 = np.log(mu2)
        c1 = (mu1 * pixels[ind][0], mu1 * pixels[ind][1], mu1 * pixels[ind][2])
        c2 = (mu2 * pixels[ind][0], mu2 * pixels[ind][1], mu2 * pixels[ind][2])
        # c1, c2 = pixels[ind], pixels[ind]
        '''
        if r1 > 10 or r1 < 1:
            circle(screen, c1, (x1, y1), 3)
        if r2 > 10 or r2 < 1:
            circle(screen, c2, (x2, y2), 3)
        else:
            w = 1
        '''

        polygon(screen, c1, [(x1 - r1,y1 + r1),(x1 - r1,y1 - r1),(x1 + r1,y1 - r1),(x1 + r1,y1 + r1)])
        polygon(screen, c2, [(x2 - r2, y2 + r2), (x2 - r2, y2 - r2), (x2 + r2, y2 - r2), (x2 + r2, y2 + r2)])
        #circle(screen, c1, (x1, y1), r)
        #circle(screen, c2, (x2, y2), r)
        # circle(screen, pixels[ind], (x0, y0), 5)


l_im = l_im_vis / 2


def from_Pz_to_sc(x):
    return x / pole_zr * sc_s + sc_s


def make_frame():
    sch = 0
    for x in np.arange(-pole_zr, pole_zr, pole_zr / sc_s):
        for y in np.arange(-pole_zr, pole_zr, pole_zr / sc_s):
            sch += 1
            i1, i2 = vision(x, y)
            with open('frame400nb.txt', 'a+') as f:
                f.write(
                    f'{from_Pz_to_sc(i1[0])} {from_Pz_to_sc(i1[1])} {i1[2]} {from_Pz_to_sc(i2[0])} {from_Pz_to_sc(i2[1])} {i2[2]}\n')
            if sch % 400 == 0:
                print(int(sch / sc_s ** 2 * 25), '%')


#make_frame()
# razdel
color = (0, 0, 0)
FPS = 1000
step = 350
# color = (255,255,255)
pygame.init()
screen = pygame.display.set_mode((2 * sc_s, 2 * sc_s))
# screen = pygame.Surface((sc_s*2, sc_s*2), pygame.SRCALPHA)
screen.fill(color)

polygon(screen, (0, 0, 255),
        [(1.1 * l_im, 1.1 * l_im), (2 * sc_s - 1.1 * l_im, 1.1 * l_im), (2 * sc_s - 1.1 * l_im, 2 * sc_s - 1.1 * l_im),
         (1.1 * l_im, 2 * sc_s - 1.1 * l_im)])

polygon(screen, color, [(1.11 * l_im, 1.11 * l_im), (2 * sc_s - 1.11 * l_im, 1.11 * l_im),
                        (2 * sc_s - 1.11 * l_im, 2 * sc_s - 1.11 * l_im), (1.11 * l_im, 2 * sc_s - 1.11 * l_im)])
circle(screen, (0, 0, 255), (sc_s, sc_s), 10)
circle(screen, (0, 255, 0), (sc_s, sc_s), revis)
circle(screen, color, (sc_s, sc_s), revis * 0.99)
circle(screen, (0, 0, 255), (sc_s, sc_s), 3)
pygame.display.update()
clock = pygame.time.Clock()

xx = sc_s
yy = sc_s
arx = []
for x in np.arange(1.05 * l_im, 2 * sc_s - 1.05 * l_im, (2 * sc_s - 2.1 * l_im) / step):
    arx.append(x)
ary = []
for y in np.arange(1.05 * l_im, 2 * sc_s - 1.05 * l_im, (2 * sc_s - 2.1 * l_im) / step):
    ary.append(y)
arx = np.array(arx)
ary = np.array(ary)
with open('frame400.txt', 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = [float(x) for x in lines[i].split()]
    for j in range(100):

        for i in range(len(arx) - 1):
            clock.tick(FPS)
            screen.fill(color)

            x0 = arx[i]
            y0 = ary[i]
            circle(screen, (255, 0, 0), (x0, y0), 3)
            par = 1
            r = ((x0 - sc_s) ** 2 + (y0 - sc_s) ** 2) ** 0.5
            '''
            circle(screen, (0, 255, 0), (sc_s, sc_s), revis)
            circle(screen, color, (sc_s, sc_s), revis * 0.99)
            circle(screen, (0, 0, 255), (sc_s, sc_s), 3)
            '''
            if r < revis * 1:
                # par = revis / (r + revis / 3)
                par = 1.5
                image_1(pixels, l, hi, x0, y0, l_im_vis, h_im_vis, lines, par)
            else:
                image_0(pixels, l, hi, x0, y0, l_im_vis, h_im_vis, lines, par)

            pygame.display.update()

pygame.quit()