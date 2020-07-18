import os, re

import numpy as np
import scipy as sp
import scipy.special as spc

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from progressbar import ProgressBar


class Harm(object):
    """Contains spherical harmonics for the dipole expansion of the MFPAD"""
    def __init__(self, p):
        self.p = p

    def fun(self, the_e, phi_e, the_n, phi_n):
            Y = spc.lpmv(self.p[3], self.p[1], np.cos(the_e))  * spc.lpmv(np.abs(self.p[4]), 
             self.p[2], np.cos(the_n)) * (self.p[0] * np.cos(self.p[3] * phi_e 
             + self.p[4] * phi_n + (self.p[5].clip(-1,0)) * 0.5 * np.pi))
            return Y

    def rffun(self, the_e, phi_e, the_n):
            Y = spc.lpmv(self.p[3], self.p[1], np.cos(the_e))  * spc.lpmv(np.abs(self.p[4]), 
             self.p[2], np.cos(the_n)) * (self.p[0] * np.cos(np.abs(self.p[3]) * phi_e ))
            return Y 

def parse_pad_dict(file):
    """Opens a dat file with the MF-PAD coefficients and returns
    a dictionary with spherical harmonics"""
    if type(file) is str:
        pad = sp.genfromtxt(file, skiprows=4)
    else:
        pad = file

    d = {}
    for i, p in enumerate(pad):
        d[i] = Harm(pad[i])
    return d

def parse_rfdat(file):
    f = np.loadtxt(file, skiprows=4)
    rf_c = f[:,0]
    rf_llp = f[:,1:]

    return rf_c, rf_llp

def angle_map(n_th, n_ph, twopi=False):
    if twopi:
        th, ph = np.linspace(0, 2*np.pi, n_th), np.linspace(0, 2* np.pi, n_ph)
    else:
        th, ph = np.linspace(0, np.pi, n_th), np.linspace(0, 2* np.pi, n_ph)
    phph, thth = np.meshgrid(ph, th)
    return thth, phph

def eval_pad(d, thth, phph, the_n, phi_n, rf=False, twopi=False):
    """Maps an MFPAD onto a spherical grid"""
    pad = np.zeros(thth.shape)
    if not rf:
        for i in d.itervalues():
            pad += i.fun(thth, phph, the_n, phi_n)

    else: 
        for i in d.itervalues():
            pad += i.rffun(thth, phph, the_n)

    return pad

def map_cart(pad, thth=np.zeros(1), phph=np.zeros(1)):
    """Maps the MFPAD from spherical to cartesian coords."""
    X, Y, Z = np.zeros(pad.shape), np.zeros(pad.shape), np.zeros(pad.shape)
    if not (thth.any() and phph.any()):
        thth, phph = angle_map(pad.shape[0], pad.shape[1])
    X = pad * np.sin(thth) * np.cos(phph)
    Y = pad * np.sin(thth) * np.sin(phph)
    Z = pad * np.cos(thth)

    return np.asarray([X, Y, Z])

def map_ang(cart_pad):
    """Maps the MFPAD back from cart. to spherical coords."""
    X, Y, Z = cart_pad
    cart_pad = np.square(cart_pad)
    R = cart_pad.sum(axis=0)
    R = np.sqrt(R)
    thth = np.arccos(Z / R)
    phph = np.arctan2(Y, X)
    return np.array([thth, phph, R])

def eulerx(pad, beta):
    eul = np.array([
        [1,                 0,            0],
        [0,      np.cos(beta), np.sin(beta)],
        [0, -1 * np.sin(beta), np.cos(beta)]
        ])
    return pad.T.dot(eul).T

def eulery(pad, beta):
    eul = np.array([
        [np.cos(beta), 0, -1 * np.sin(beta)],
        [           0, 1,                 0],
        [np.sin(beta), 0,      np.cos(beta)]
        ])
    return pad.T.dot(eul).T

def eulerz(pad, alpha):
    eul = np.array([
        [     np.cos(alpha), np.sin(alpha), 0],
        [-1 * np.sin(alpha), np.cos(alpha), 0],
        [                 0,             0, 1]
        ])
    return pad.T.dot(eul).T

def plot_pad(cart_pad):
    X, Y, Z = cart_pad
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(-5,5)
    ax.set_ylim3d(-5,5)
    ax.set_zlim3d(-5,5)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='#74abee', linewidth=0)

    return surf


def avg_pad_rf(d, weights, n_th, n_ph, n_bet=0):
    pad_out = np.zeros(( n_th, n_ph))
    al, bet = np.linspace(0, 2*np.pi, n_bet), np.linspace(0, np.pi, n_th)
    angs, unit = angle_map(n_th, n_ph), np.ones((n_th, n_ph))

    angs_c = map_cart(unit, angs[0], angs[1])
    for i, b in enumerate(bet):
            angs_cr = eulery(angs_c, b)
            thth_r, phph_r = map_ang(angs_cr)[:2]
            pad = eval_pad(d, thth_r, phph_r,  -1 * b, 0, rf=True)
            pad_out += pad * weights[i] * np.sin(b)

    th_pad = pad_out / n_th
    pad_out[:,:] = 0
    for j in np.arange(n_ph):
        ph_pad = np.roll(th_pad, j, axis=1)
        pad_out += ph_pad
    return th_pad, pad_out / n_ph

def axis_dist(the, sig):
    return np.exp(-0.5 * np.sin(the)**2 / sig**2)
def get_cos2(sig):
    def f_up(th, sig):
        return np.sin(th) * np.cos(th) ** 2 * axis_dist(th, sig) ** 2
    def f_lo(th, sig):
        return np.sin(th) * axis_dist(th, sig) ** 2

    return sp.integrate.quad(f_up, 0, np.pi/2, args=sig)[0] / sp.integrate.quad(f_lo, 0, np.pi/2, args=(sig))[0]

if __name__ == '__main__':
    file='exp3.dat'
    pad = sp.genfromtxt(file, skiprows=4,
#       dtype=[(np.float64, np.int64, np.int64, np.int64, np.int64, np.int64)]
        )

    d = parse_pad_dict(pad)
#   c = eval_pad(d, 100, 50, 0,0, rf=True)
#   m = map_cart(c)
#   plot_pad(m)
#
#   c = eval_pad(d, 100, 50, np.pi/2,0, rf=True)
#   m = map_cart(c)
#   plot_pad(m)
#   plt.show()
