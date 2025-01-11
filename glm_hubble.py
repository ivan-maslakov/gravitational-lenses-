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

def Hub_dist(z):
    return 300000/70*(z**2 + 2*z)/(z**2+2*z+2)

x = np.arange(0, 10, 0.01)
fig3 = plt.figure(figsize=(7, 6))
ax = fig3.add_subplot()
ax.scatter(x, d_ang(0,x, 70),s = 2, color = 'b', label = 'd_ang')
ax.scatter(x, Hub_dist(x),s = 2, color = 'g', label = 'Hubble`s law')
ax.set_xlabel('z')
ax.set_ylabel('D, Mps')

plt.grid()
plt.legend()
plt.show()
