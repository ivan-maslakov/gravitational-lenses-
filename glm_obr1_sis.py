import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.draw import *
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
#from numba import jit

im = Image.open('obr4.jpg')
pr_color = 35
rmk = 1/5
r = 1
g = 1
b = 1
horizontal = True
#horizontal = False

class layer:
    def __init__(self, beg, end, numb):
        self.beg = beg
        self.end = end

        self.n = numb
        self.oblast = 0
        self.neu = set()
    def layer_conect(self, ly):
        if ((self.beg >= ly.beg and self.beg <= ly.end) or (self.end >= ly.beg and self.end <= ly.end)) and abs(ly.n - self.n) == 1:
            ly.neu.add(self)
            self.neu.add(ly)
            return True
        else:
            return False

l, hi = im.size
pixels = im.load() # create the pixel map
pix0 = []

for i in range(im.size[0]): # for every pixel:
    for j in range(im.size[1]):
        pix0.append([[i,j],pixels[i, j]])
def find_center_zasvetky():
    mc = 0
    for p in (pix0):
        if p[1][0] + p[1][1] + p[1][2] > mc:
            mc = p[1][0] + p[1][1] + p[1][2]
    c = np.array([0, 0])
    mc = float(mc)
    sc = 0
    for p in (pix0):
        sc+=1.0 ** (1 * (p[1][0] + p[1][1] + p[1][2] - mc))
        p[0][0] = float(p[0][0])
        p[0][1] = float(p[0][1])
        c = c + np.array(p[0]) * (1.0 ** (1 * (p[1][0] + p[1][1] + p[1][2] - mc)))
    c = c / sc
    c[0] = int(c[0])
    c[1] = int(c[1])
    return c


def take_center_zasvetky():
    pygame.init()
    screen = pygame.display.set_mode((l, hi))
    screen.fill((0, 0, 0))
    clock = pygame.time.Clock()
    for p in pix0:
        circle(screen, p[1], (p[0][0], p[0][1]), 1)
    finished = False
    schpg = 0
    crds = []
    while not finished and schpg < 1:
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


def point(a, color):
    for i in range(l):
        for j in range(hi):
            if ((i - a[0]) ** 2 + (j - a[1]) ** 2) ** 0.5 < 2:
                pixels[i, j] = color
def point2(a, color):
    pixels[a[0], a[1]] = color
    for i in np.arange(-1, 1):
        for j in np.arange(-1, 1):
            pixels[a[0] + i, a[1] + j] = color

def dist(p1, p2):
    return np.dot((p1 - p2), (p1 - p2)) ** 0.5
def conv1(x, y):
    #из нормальных в пиллоу
    x = x + l // 2
    y = hi // 2 - y
    return (x, y)

def conv2(x, y):
    x = x - l // 2
    y = hi // 2 - y
    return (x, y)

#center = take_center_zasvetky()
center = find_center_zasvetky()
point(center, (255, 0, 0))
#im.show()
pix = []
arr = []
arrr = []
arrg = []
arrb = []
'''
for i in range(im.size[0]): # for every pixel:
    for j in range(im.size[1]):
        p = pixels[i, j]
        arr.append(p[0] + p[1] + p[2])
        arrr.append(p[0])
        arrg.append(p[1])
        arrb.append(p[2])


arx = np.arange(0, 255, 5)


ary = []
aryr = []
aryg = []
aryb = []
for i in arx:
    ans = 0
    for j in arr:
        if j > i:
            ans += 1
    ary.append(ans)
    ans = 0
    for j in arrr:
        if j > i:
            ans += 1
    aryr.append(ans)
    ans = 0
    for j in arrg:
        if j > i:
            ans += 1
    aryg.append(ans)
    ans = 0
    for j in arrb:
        if j > i:
            ans += 1
    aryb.append(ans)
'''

fig3 = plt.figure(figsize=(10, 10))
ax3 = fig3.add_subplot()
#ax4 = fig3.add_subplot(3, 1, 2)
#ax5 = fig3.add_subplot(3, 1, 3)




arx1 = []
ary1 = []
for i in range(min(im.size[0], im.size[1])):
    arx1.append(i)
    ary1.append(r * pixels[i, hi // 2][0] + g * pixels[i, hi // 2][1] + b * pixels[i, hi // 2][2])
arx2 = []
ary2 = []
for i in range(min(im.size[1], im.size[0])):
    arx2.append(i)
    ary2.append(r * pixels[l // 2, i][0] + g * pixels[l // 2, i][1] + b * pixels[l // 2, i][2])
#ОБРЕЗАЕМ

def sred(a, par):
    a = list(a)
    n_a = a[:par]

    for i in range(par, len(a) - par):
        new_el = (np.array(a[i - par:i + par]).sum()) / 2 / par
        n_a.append(new_el)
    for i in range(len(a) - par, len(a)):
        n_a.append(a[i])
    return n_a
def draw_cline(center, k):
    lolya = max(l, hi)
    x0 = center[0]
    y0 = center[1]
    for i in np.arange(-lolya, lolya, lolya / 1000):
        x = int(i) + x0
        y = y0 + int(k * i)
        if 0 < x < l and 0 < y < hi:
            pixels[x, y] = (0, 255, 0)
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

profile = take_profile(center, horizontal)
without_profile(profile, center)
im.show()
#ax3.scatter(arx1, ary2, color = 'black')
#print(len(arx2))
ary2 = sred(ary2, 5)
ax3.scatter(arx1, ary2, s = 3, color = 'g', label = 'ver. pr.')
#ax3.scatter(arx2, ary1, color = 'r', label = 'ver. pr.')
ar_copy = ary1
ary1 = sred(ary1, 5)
ax3.scatter(arx2, ary1, s = 3, color = 'b', label = 'Hor. pr.')
ary1 = np.array(ary1)
ary2 = np.array(ary2)
ary = sred(ary2 - ary1, 1)
ax3.scatter(arx2, ary, s = 3, color = 'r', label = 'Norm. pr.')
#draw_cline([l//2,hi//2], 0)
#draw_cline([l//2,hi//2], 25)

#ax4.grid()
#ax4.legend()
#ax5.grid()
#ax5.legend()
ax3.set_xlabel('r, pix')
ax3.set_ylabel('rel. flux, r.u.')
ax3.grid()
ax3.legend()
for i in range(im.size[0]): # for every pixel:
    for j in range(im.size[1]):
        pixels[i, j] = (int(r * pixels[i, j][0]), int(g * pixels[i, j][1]), int(b * pixels[i, j][2]))
for i in range(im.size[0]): # for every pixel:
    for j in range(im.size[1]):
        #rmk = 1/4
        prr_color = pr_color * (r + g + b) / 3
        #obr1 - 200
        if (2 * pixels[i, j][0] + 1 * pixels[i, j][1] + 1 * pixels[i, j][2]) < prr_color or (abs(i - im.size[0] / 2) / im.size[0] > rmk or abs(j - im.size[1] / 2) / im.size[1] > rmk):
            pixels[i, j] = (0, 0 ,0)
        #elif pixels[i, j][2] < 0:
            #pixels[i, j] = (0, 0, 0)
        else:
            x, y = conv2(i, j)
            pix.append([np.array([i, j]), pixels[i, j]])
#8arcsec
#im.show()
#plt.show()

pygame.init()
screen = pygame.display.set_mode((l, hi))
screen.fill((0,0,0))
clock = pygame.time.Clock()
for p in pix:
    circle(screen, (255, 0, 255), (p[0][0], p[0][1]), 1)
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

k_ang = ((crds[0] - crds[2]) / (crds[1] - crds[3])) ** -1
b_lin = crds[1] - k_ang * crds[0]
x1 = crds[0]
y1 = crds[1]
x2 = crds[2]
y2 = crds[3]
print(crds)
#obr1 - 0.1
'''
for i in np.arange(0, l, 1):
    j = k_ang * i + b_lin
    if abs(i) < l-10 and abs(j) < hi-10:

        pixels[i, j] = (0, 0, 255)
'''
#pixels[l // 2, hi // 2] = (0, 0, 255)
im1 = []
im2 = []

#ДЕЛИМ НА ДВЕ ДУГИ
def obhod(im):
    layers = set()
    for j in range(im.size[1]):
        in_lay = False
        #cur_end = 0
        for i in range(im.size[0]):
            if pixels[i, j] != (0,0,0):
                if not in_lay:
                    #cur_end += 1

                    cur_lay = layer(i, i, j)
                    in_lay = True
            elif in_lay:
                in_lay = False
                cur_lay.end = i - 1
                layers.add(cur_lay)
    return layers

layers = obhod(im)

#for i in layers:
    #print(i.beg, i.end, i.n)

for lay in layers:
    if lay.n == y1:
        if lay.beg <= x1 <= lay.end:
            lay1 = lay
            lay1.oblast = 1
    if lay.n == y2:
        if lay.beg <= x2 <= lay.end:
            lay2 = lay
            lay2.oblast = 2


obl1 = []
obl2 = []
obl1.append(lay1)
obl2.append(lay2)
lpr1 = 0
lpr2 = 0

def neub(layers):
    for l1 in layers:
        for l2 in layers:
            l1.layer_conect(l2)
            l2.layer_conect(l1)

neub(layers)

def obl(obl, lpr, layers):
    stp = False
    while not stp:
        if lpr == len(obl):
            stp = True
        lpr = len(obl)
        for layer in obl:
            for neu in layer.neu:
                if neu not in obl:
                    obl.append(neu)
    return obl
obl1 = obl(obl1, lpr1, layers)
#print('z1')
obl2 = obl(obl2, lpr2, layers)
#print('z2')

def napoln_im(obl, pixels):
    im = []
    for layer in obl:
        for i in range(layer.beg, layer.end + 1):
            im.append([np.array([i, layer.n]), pixels[i, layer.n]])
    return im
im1 = napoln_im(obl1, pixels)
im2 = napoln_im(obl2, pixels)





'''

for p in pix:
    if p[0][1] < p[0][0] * k_ang + b_lin:
        im1.append(p)
    else:
        im2.append(p)
'''
#print(len(im1))
#print(len(im2))
#im.show()
'''
for p in im1:
    pixels[p[0][0], p[0][1]] = (0, 255, 0)
im.show()
'''
#ИЩЕМ КОНЦЫ

def endls(imm):
    c1 = np.array([0, 0])
    for p in imm:
        #pixels[p[0][0], p[0][1]] = (0,0,0)
        c1 = c1 + p[0]
    c1 = c1 / len(imm)
    #pixels[c1[0], c1[1]] = (0, 255, 0)
    #point([c1[0], c1[1]], (0, 255, 255))
    max_dist = 0
    k11 = c1
    for p in imm:
        d = dist(c1, p[0])
        if d > max_dist:
            max_dist = d
            k11 = p[0]
    #pixels[k11[0], k11[1]] = (255, 0, 0)
    #point(np.array([k11[0], k11[1]]), (255, 255, 255))
    d1c = max_dist
    max_dist = 0

    k12 = c1
    for p in imm:
        d1 = dist(k11, p[0])
        dc = dist(c1, p[0])
        if dc > max_dist and d1 > d1c:
            max_dist = dc
            k12 = p[0]
    #point(np.array([k12[0], k12[1]]), (255, 255, 255))
    return k11, k12
k11, k12 = endls(im1)
k21, k22 = endls(im2)

#ИЩЕМ ЛИНЗУ


if np.dot(k11 - k12, k11 - k12) < np.dot(k21 - k22, k21 - k22):
    k11, k12, k21, k22, im1, im2 = k21, k22, k11, k12, im2, im1
def lc_coords(k11, k12, k21, k22):
    k1 = (k11[1] - k12[1]) / (k11[0] - k12[0])
    k2 = (k21[1] - k22[1]) / (k21[0] - k22[0])
    b1 = k11[1] - k11[0] * k1
    b2 = k21[1] - k21[0] * k2
    xc0 = (b2 - b1) / (k1 - k2)
    yc0 = k1 * xc0 + b1
    c0_ = np.array([xc0, yc0])
    if dist(np.array([xc0, yc0]), k11) > dist(k11, k12) or dist(np.array([xc0, yc0]), k12) > dist(k11, k12):
        return lc_coords(k11, k22, k21, k12)
    else:
        return c0_
c0 = lc_coords(k11, k21, k12, k22)
#print('lense', c0)
#pixels[c0[0], c0[1]] = (255, 255, 255)
#point(np.array([c0[0], c0[1]]), (255, 255, 255))
def find_pare(p_, imm_, c0):
    k_ = (p_[0][1] - c0[1]) / (p_[0][0] - c0[0])
    b_ = c0[1] - k_ * c0[0]
    md_ = 10000000
    pare_ = [[50,50], (0,0,0)]
    ind = 0
    for ppp in imm_:
        if abs((ppp[0][1] - ppp[0][0] * k_ - b_) / l) < 0.003:
            #print('w')
            d_ = dist(ppp[0], c0)
            if md_ > d_:
                #print('c')
                md_ = d_
                pare_ = ppp
                ind += 1
                #point2([pare_[0][0], pare_[0][1]], (255, 255, 255))
                #print(pare_[0])
    #print(pare_[0])
    if ind <= 0:
        #print('error')
        return [0, 0]
    return pare_[0]



#ИЩЕМ РАДИУС ЭЙНШТЕЙНА
def find_te(im1, im2, c0, pogr):
    te = 0
    beta1 = 0
    beta2 = 0
    vsp = 100
    vspp = vsp
    for i in range(vsp * 0, vsp):
        indf = 1
        indff = 0
        p1 = im1[i * (len(im1) // vsp)]
        #point2([p1[0][0],p1[0][1]],(255, 255, 255))
        k = (p1[0][1] - c0[1]) / (p1[0][0] - c0[0])
        b = c0[1] - k * c0[0]
        md = 0
        pare = p1

        for p_p in im1:
            #print(c0)
            if abs((p_p[0][1] - p_p[0][0] * k - b) / l) < 0.003:
                d = dist(p_p[0], c0)
                if md < d:
                    md = d
                    pare = p_p
                    indff += 1
                    #point2([pare[0][0],pare[0][1]],(255, 255, 255))
        p1 = pare
        p2 = find_pare(p1, im2, c0)
        if indff == 0:
            #print('er1')
            indf = 0
        if p2[0] == 0 and p2[1] == 0:
            indf = 0
        if indf == 1:
            #point2(p1[0], (0,255,0))
            #point2(p2, (0, 255, 0))
            #pixels[p1[0][0], p1[0][1]] = (0, 255, 0)
            #pixels[p2[0], p2[1]] = (255, 255, 255)
            r1 = dist(p1[0], c0)
            #print(*p2)
            r2 = dist(p2, c0)
            te = te + (r1 + r2) / 2
            pogr.append((r1 + r2) / 2)
            beta2 = beta2 + (r1 + r2) / 2
            beta1 = beta1 + (r1 * r2) ** 0.5
        else:
            vspp -= 1
    print(beta1, beta2, te / vspp)

    return te / vspp, beta1 / vspp, beta2 / vspp, pogr
pogr = []
te, beta1, beta2, pogr = find_te(im1, im2, c0, pogr)
pogr = np.array(pogr)
sr = pogr.sum() / len(pogr)
sig = (((pogr - sr) ** 2).sum() / len(pogr))**0.5
print('SIGMA', sig ,'TETA_EIN', te)
#im.show()

for i in range(l):
    for j in range(hi):
        if te - 1 < ((i - c0[0]) ** 2 + (j - c0[1]) ** 2) ** 0.5 < te + 1:
            pixels[i, j] = (0, 255, 0)
#im.show()
'''
for i in range(im.size[0]): # for every pixel:
    for j in range(im.size[1]):
        pixels[i, j] = (0, 0 ,0)
'''

for i in range(im.size[0]): # for every pixel:
    for j in range(im.size[1]):
        pixels[i, j] = (0,0,0)

for p in im1:
    pixels[p[0][0], p[0][1]] = (0, 255, 255)
    #pixels[p[0][0], p[0][1]] = (0, 0, 0)
#im.show()
for p in im2:
    pixels[p[0][0], p[0][1]] = (255, 255, 0)
    #pixels[p[0][0], p[0][1]] = (0, 0, 0)
#im.show()
for i in range(l):
    for j in range(hi):
        if te - 1 < ((i - c0[0]) ** 2 + (j - c0[1]) ** 2) ** 0.5 < te + 1:
            pixels[i, j] = (0, 255, 0)
def obr2():
    for p in im2:
        x = p[0][0] - c0[0]
        y = p[0][1] - c0[1]

        r = dist(c0, p[0])
        #print(r, te)
        r0 = r - te ** 2 / r
        #r0 = r - te
        #print(r0)
        sin = y / (x ** 2 + y ** 2) ** 0.5
        cos = x / (x ** 2 + y ** 2) ** 0.5
        i = c0[0] + r0 * cos
        j = c0[1] + r0 * sin
        #print(i, j)
        if abs(i) < l and abs(j) < hi:
            #pixels[i, j] = p[1]
            pixels[i, j] = (0, 155, 155)
        #print('col', p[1])
def obr1():
    for p in im1:
        x = p[0][0] - c0[0]
        y = p[0][1] - c0[1]

        r = dist(c0, p[0])
        # print(r, te)
        r0 = r - te ** 2 / r
        #r0 = r - te
        # print(r0)
        sin = y / (x ** 2 + y ** 2) ** 0.5
        cos = x / (x ** 2 + y ** 2) ** 0.5
        i = c0[0] + r0 * cos
        j = c0[1] + r0 * sin
        # print(i, j)
        if abs(i) < l and abs(j) < hi:
            # pixels[i, j] = p[1]
            pixels[i, j] = (0, 155, 155)
            # print('col', p[1])
def obr_sis_1():
    for p in im1:
        x = p[0][0] - c0[0]
        y = p[0][1] - c0[1]

        r = dist(c0, p[0])
        # print(r, te)
        r0 = r - te
        # print(r0)
        sin = y / (x ** 2 + y ** 2) ** 0.5
        cos = x / (x ** 2 + y ** 2) ** 0.5
        i = c0[0] + r0 * cos
        j = c0[1] + r0 * sin
        # print(i, j)
        if abs(i) < l and abs(j) < hi:
            # pixels[i, j] = p[1]
            pixels[i, j] = (0, 155, 155)
            # print('col', p[1])
def obr_sis_2():
    for p in im2:
        x = p[0][0] - c0[0]
        y = p[0][1] - c0[1]

        r = dist(c0, p[0])
        #print(r, te)
        r0 =r - te
        #print(r0)
        sin = y / (x ** 2 + y ** 2) ** 0.5
        cos = x / (x ** 2 + y ** 2) ** 0.5
        i = c0[0] + r0 * cos
        j = c0[1] + r0 * sin
        #print(i, j)
        if abs(i) < l and abs(j) < hi:
            #pixels[i, j] = p[1]
            pixels[i, j] = (155, 155, 155)
        #print('col', p[1])
obr_sis_1()
obr_sis_2()
'''
for p in im2:
    x = p[0][0] - c0[0]
    y = p[0][1] - c0[1]

    r = dist(c0, p[0])
    #print(r, te)
    r0 = -r + te ** 2 / r
    #print(r0)
    sin = y / (x ** 2 + y ** 2) ** 0.5
    cos = x / (x ** 2 + y ** 2) ** 0.5
    i = c0[0] + r0 * cos
    j = c0[1] + r0 * sin
    #print(i, j)
    if abs(i) < l and abs(j) < hi:
        pixels[i, j] = p[1]
'''

im.show()

print(l, hi)
def mu(x):
    mu = (1 - x ** -4)
    if mu > 0:
        return mu
    else:
        return 1 / abs(mu)

zl = 0.729
zs = 2.32
x = np.arange(0, l, 1)
y = np.arange(0, hi, 1)
c = 300000000
dl = 965.5 * 10 ** 6 * 206265 * 150000000000
ds  = 1359 * 10 ** 6 * 206265 * 150000000000
dls = 552 * 10 ** 6 * 206265 * 150000000000
M = 2 * 10 ** 30 * 10 ** 11.42
G = 6.67 * 10**(-11)
Rc = 2 * G * M / c ** 2
pole_zr = 8.28

par_c = te / l * pole_zr
cli = 300000000
#par_c = 1
beta1 = beta1 / l * pole_zr
beta2 = beta2 / l * pole_zr
def delta_t(zd, rs, teta0, teta2):
    #print(teta2)
    return (1 + zl) / cli * ds * dl / dls * ((teta0 - teta2) * (teta0 + teta2 - 2 * beta1) / 2 / 206265 ** 2 - 4 * G * M / c ** 2 * dls / dl / ds * np.log(teta0 / teta2))
tee = te / l * pole_zr
def delta_t_sis(zd, rs, teta0, teta2):

    #tee = te / l * pole_zr
    #print(teta22)
    return (1 + zl) / cli * ds * dl / dls / 206265 ** 2 * ((teta0 - teta2) * (teta0 + teta2 - 2 * beta2) / 2 - tee * (teta0 - teta2))

color_set = set()
for p in im1:
    xcs = p[0][0]
    ycs = p[0][1]
    color_set.add((xcs, ycs))
for p in im2:
    xcs = p[0][0]
    ycs = p[0][1]
    color_set.add((xcs, ycs))
xxx = []
yyy = []
zzz = []
xxxs = []
yyys = []
zzzs = []
xxxx = []
yyyy = []
zzzz = []
cxx = c0[0] / l * pole_zr
cyy = c0[1] / hi * pole_zr
for i in x:
    xx = i / l * pole_zr
    for j in y:
        yy = j / hi * pole_zr
        dt = delta_t(zl, Rc, par_c, ((xx - cxx) ** 2 + (yy - cyy) ** 2) ** 0.5)
        dts = delta_t_sis(zl, Rc, par_c, ((xx - cxx) ** 2 + (yy - cyy) ** 2) ** 0.5)
        if i == int(c0[0]) and j == int(c0[1]):
            print(dt, dts)
        xxx.append(i)
        yyy.append(j)
        zzz.append(dt)
        xxxs.append(i)
        yyys.append(j)
        zzzs.append(dts)
        if (i, j) in color_set:
            yy = j / hi * pole_zr
            dt = delta_t(zl, Rc, par_c, ((xx - cxx) ** 2 + (yy - cyy) ** 2) ** 0.5)
            xxxx.append(i)
            yyyy.append(j)
            zzzz.append(dt)


#print(*xxx)
#print(*yyy)
#print(*zzz)
step_warframe = 20
fig55 = plt.figure()
ax55 = fig55.add_subplot(111, projection='3d')

ax55.scatter(xxx[::step_warframe], yyy[::step_warframe], zzz[::step_warframe], color = 'b', alpha = 0.2,s = 1)
ax55.scatter(xxxs[::step_warframe], yyys[::step_warframe], zzzs[::step_warframe], color = 'r', alpha = 0.2,s = 1)
#ax55.scatter(xxxx[::step_warframe], yyyy[::step_warframe], zzzz[::step_warframe], color = 'b', s = 2)

ax55.set_xlim([min(xxx), max(xxx)])
ax55.set_ylim([min(yyy), max(yyy)])
ax55.set_zlim([min(zzz), max(zzz)])
ax55.set_xlabel('x, r.u.')
ax55.set_ylabel('y, r.u.')
ax55.set_zlabel('time_delay, r.u.')
plt.show()

