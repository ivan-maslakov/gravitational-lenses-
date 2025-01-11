import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.draw import *
from PIL import Image
im = Image.open('radio.jpg')

pixels = list(im.getdata())
l, hi = im.size


def d_ang(z1, z2, H0, Omega_m = 0.3, Omega_la = 0.7):
    #print(300000 / (1 + z2) * integral(h_pod_int, z1, z2))
    #print(300000 / (1 + z2) * integral(h_pod_int, z1, z2))
    return 300000 / (1 + z2) * integral(h_pod_int, z1, z2, H0)


def h_pod_int(z, H0, Omega_m = 0.3, Omega_la = 0.7):
    return 1 / h(z, H0)

def h(z, H0, Omega_m = 0.3, Omega_la = 0.7):
    return H0 * (Omega_m * (1 + z) ** 3 + Omega_la) ** 0.5

def integral(f, x1, x2, H0):
    tochn = 10000
    if x1 != 0:
        step = min(x1 / tochn, (x2 - x1) / tochn)
    else:
        step = (x2 - x1) / tochn
    ans = 0
    for i in range(tochn):
        ans += step * f(x1 + i * step, H0)
    return ans
#z11 = 0.355
#z22 = 1.41
zl = 0.729
zs = 2.32
teta0 = 2.138 / 206265
teta2 = 1.16 / 206265
#print(d_ang(0,z11))
#print(d_ang(0,z22))
#print(d_ang(z11,z22))
x = np.arange(0, l, 1)
y = np.arange(0, hi, 1)
c = 300000000
G = 6.67 * 10**(-11)
hs = []
sis = []
points = []
for i in np.arange(37,120,1):

    dl = d_ang(0,zl, i) * 10 ** 6 * 206265 * 150000000000
    ds  = d_ang(0,zs, i) * 10 ** 6 * 206265 * 150000000000
    dls = d_ang(zl,zs, i) * 10 ** 6 * 206265 * 150000000000
    #M = 5.66 * 10 ** 42
    tee2 =(teta0 * teta2) ** 0.5
    tee1 =(teta0 + teta2) * 0.5
    #print('te_sis', tee1 * 206265)
    #print(tee1, tee2)
    beta1 = (teta0 - teta2) * 0.5
    #print('beta_sis', beta1 * 206265)
    beta2 = teta0 - tee2 ** 2 /teta0
    #print(-teta2 + tee2 ** 2 /teta2, beta2)
    M = (teta0 - beta2) / 4 / G / dls * c ** 2 * dl * teta0 * ds
    G = 6.67 * 10**(-11)
    Rc = 2 * G * M / c ** 2
    #pole_zr = 8.28
    zl = 0.355
    par_c = 0.1 / 206265
    cli = 300000000
    #par_c = 1

    #beta2 = 4.11
    #teta0 = 5.27
    #teta2 = 1.16
    #tee = 2.47
    if i == 70:
        y1 = []
        y2 = []
        x = []
        for j in np.arange(tee2 / 100,2 * tee2, tee2 / 100):

            y2.append(-1 * ((1 + zl) / cli * ds * dl / dls* ((teta0 - j) * (teta0 + j - 2 * beta1) / 2 - tee1 * (teta0 - j))) / 24 / 3600)
            x.append(j)
            y1.append(-1 * (1 + zl) / cli * ds * dl / dls * ((teta0 - j) * (teta0 + j - 2 * beta2) / 2 - 4 * G * M / c ** 2 * dls / dl / ds * np.log(teta0 / j)) / 24 / 3600)
    hs.append(i)
    points.append(-1 * (1 + zl) / cli * ds * dl / dls * ((teta0 - teta2) * (teta0 + teta2 - 2 * beta2) / 2 - 4 * G * M / c ** 2 * dls / dl / ds * np.log(teta0 / teta2)) / 24 / 3600)
    sis.append(-1 * ((1 + zl) / cli * ds * dl / dls* ((teta0 - teta2) * (teta0 + teta2 - 2 * beta1) / 2 - tee1 * (teta0 - teta2))) / 24 / 3600)

fig3 = plt.figure(figsize=(7, 6))
ax = fig3.add_subplot()
ax.scatter(hs, sis, s = 3, color = 'r', label = 'sis')
ax.scatter(hs, points, s = 3, color = 'b', label = 'point')
x = np.arange(37,120,0.1)
y1 = x / x * (140 + 10)
y2 = x / x * (140 - 10)
xc = np.arange(37.5,120.5,1)
yc = xc/xc * 152
ax.scatter(x,y1,s=1, color = 'g')
ax.scatter(x,y2,s=1, color = 'g', label = 'calculated time delay (140 days)')
#ax.scatter(xc,yc,s=2, color = 'g', label = 'calculated time delay (152 days)')
plt.grid()
plt.legend()
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot()
ax1.scatter(x, y1, s = 3, color = 'b')
ax1.scatter(x, y2, s = 3, color = 'r')

ax1.set_xlabel('theta, Rad')
ax1.set_ylabel('dt, days')
ax.set_xlabel('H0, km/s/Mpc')
ax.set_ylabel('dt, days')

plt.legend()
plt.grid()
plt.show()
