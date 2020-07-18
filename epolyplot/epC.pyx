import numpy as np
cimport numpy as np
from math import cos, sin
from scipy.special import sph_harm, lpmv
import bottleneck as bn

cdef np.ndarray angle_map(int n_th, int n_ph):
    cdef np.ndarray th = np.linspace(0, np.pi, n_th), 
    cdef np.ndarray ph = np.linspace(0, 2* np.pi, n_ph)
    phph, thth = np.meshgrid(ph, th)
    return np.array([thth, phph])

cdef np.ndarray map_cart(np.ndarray pad, 
                         np.ndarray thth=np.zeros(1), 
                         np.ndarray phph=np.zeros(1)):
    """Maps the MFPAD from spherical to cartesian coords."""
    cdef np.ndarray X 
    cdef np.ndarray Y   
    cdef np.ndarray Z 

    if not (thth.any() and phph.any()):
        thth, phph = angle_map(pad.shape[0], pad.shape[1])
    X = pad * np.sin(thth) * np.cos(phph)
    Y = pad * np.sin(thth) * np.sin(phph)
    Z = pad * np.cos(thth)

    return np.asarray([X, Y, Z])

cdef np.ndarray map_ang(np.ndarray cart_pad):
    """Maps the MFPAD back from cart. to sperical coords."""
    cdef np.ndarray X,Y,Z,R,thth,phph
    X, Y, Z = cart_pad
    cart_pad = np.square(cart_pad)
    R = bn.nansum(cart_pad, 0)
    R = np.sqrt(R)
    thth = np.arccos(Z / R)
    phph = np.arctan2(Y, X)
    return np.array([thth, phph, R])

cdef np.ndarray eulery(np.ndarray pad, double beta):
    eul = np.array([
        [cos(beta), 0, -1 * sin(beta)],
        [0, 1, 0],
        [sin(beta), 0, cos(beta)]
        ])
    return pad.T.dot(eul).T
cdef np.ndarray eulerz(np.ndarray pad, double alpha):
    eul = np.array([
        [cos(alpha), sin(alpha), 0],
        [-1 * sin(alpha), cos(alpha), 0],
        [0, 0, 1]
        ])
    return pad.T.dot(eul).T

#===============================================
cpdef np.ndarray eval_pad(np.ndarray coeffs,
                         np.ndarray LlpMmp, 
                         np.ndarray thth, 
                         np.ndarray phph, 
                         double th_n, 
                         double ph_n):
    cdef np.ndarray pad_p, pad_mol, fac
    ph_n = 2*np.pi - ph_n
    pad_p = sph_harm(LlpMmp[:,3][:,None,None], LlpMmp[:,1][:,None,None]
                     , ph_n, th_n)
    pad_mol = sph_harm(LlpMmp[:,2][:,None,None], LlpMmp[:,0][:,None,None]
         , phph[None,:,:], thth[None,:,:]) 
    fac = coeffs[:,None,None] #*((-1) ** (LlpMmp[:,2]) * 
#                                (-1) ** (LlpMmp[:,3]))[:,None,None]
    return np.sum(pad_p * pad_mol * fac,0)
#===============================================
cpdef np.ndarray eval_pad_real(np.ndarray coeffs,
                         np.ndarray LlpMmp, 
                         np.ndarray thth, 
                         np.ndarray phph, 
                         double th_n, 
                         double ph_n):
    
    cdef np.ndarray pad_p, pad_mol, pad_azi
    L, Lp = LlpMmp[:,0][:,None,None], LlpMmp[:,1][:,None,None]
    M, Mp = LlpMmp[:,2][:,None,None], LlpMmp[:,3][:,None,None]
    phase = LlpMmp[:,4]
    phase = (0.5 * np.pi * phase.clip(-1,0))[:,None,None]
    thth = np.cos(thth)
    
    pad_p = lpmv(np.abs(Mp), Lp, cos(th_n))
    pad_mol = lpmv(M, L, thth[None,:,:])
    pad_azi = coeffs[:,None,None] * np.cos(M * phph[None,:,:]
                                           + Mp * ph_n + phase)
    return np.sum(pad_p * pad_mol * pad_azi,0)
#===============================================================================
cpdef np.ndarray average_pad(np.ndarray coeffs,
                np.ndarray LlpMmp, 
                np.ndarray weights,
                int n_th, int n_ph, 
                int n_gam):
    
    cdef int i, j, k
    cdef double d_gam
    cdef np.ndarray alph, bet, gam, sinbe
    cdef np.ndarray angs_c, pad_out, angs_cr, thth_r, phph_r, pad_r
    
#    alph = np.linspace(0,2*np.pi,n_alph)
    bet = np.linspace(0,np.pi,weights.shape[0])
    gam = np.linspace(0,2*np.pi,n_gam)
    d_gam = 2 * np.pi / (n_gam-1)
    sinbe = np.sin(bet)
    angs, unit = angle_map(n_th, n_ph), np.ones((n_th,n_ph))
    angs_c = map_cart(unit, angs[0], angs[1])
    pad_out=np.zeros((n_th, n_ph), dtype=np.complex_)
#    for i in range(n_alph):
#        angs_cr = eulerz(angs_c, alph[i])
    for j in range(len(bet)):
        angs_be = eulery(angs_c, bet[j])
        for k in range(n_gam-1):
            angs_cr = eulerz(angs_be,gam[k])
            thth_r, phph_r = map_ang(angs_cr)[:2]
            pad_r = eval_pad(coeffs, LlpMmp, thth_r, phph_r, 
                             bet[j], gam[k])
            pad_out += pad_r *sinbe[j] * weights[j]
    th_pad = pad_out / (n_th*(n_gam-1))
    pad_out[:,:] = 0.
            
    for i in range(pad_out.shape[1]):
        ph_pad = np.roll(th_pad, i, axis=1)
        pad_out += ph_pad
            
    return pad_out / pad_out.shape[1]
#==================================================
#==================================================
cpdef np.ndarray average_pad_real(np.ndarray coeffs,
                np.ndarray LlpMmp, 
                np.ndarray weights,
                int n_th, int n_ph, 
                int n_gam):
    
    cdef int i, j, k
    cdef double d_gam
    cdef np.ndarray alph, bet, gam, sinbe
    cdef np.ndarray angs_c, pad_out, angs_cr, thth_r, phph_r, pad_r
    
#    alph = np.linspace(0,2*np.pi,n_alph)
    bet = np.linspace(0,np.pi,weights.shape[0])
    gam = np.linspace(0,2*np.pi,n_gam)
    d_gam = 2 * np.pi / (n_gam-1)
    sinbe = np.sin(bet)
    angs, unit = angle_map(n_th, n_ph), np.ones((n_th,n_ph))
    angs_c = map_cart(unit, angs[0], angs[1])
    pad_out=np.zeros((n_th, n_ph), dtype=np.float_)
#    for i in range(n_alph):
#        angs_cr = eulerz(angs_c, alph[i])
    for j in range(len(bet)):
        angs_be = eulery(angs_c, bet[j])
        for k in range(n_gam-1):
            angs_cr = eulerz(angs_be,gam[k])
            thth_r, phph_r = map_ang(angs_cr)[:2]
            pad_r = eval_pad_real(coeffs, LlpMmp, thth_r, phph_r, 
                             bet[j], gam[k])
            pad_out += pad_r *sinbe[j] * weights[j]
    th_pad = pad_out / (n_th*(n_gam-1))
    pad_out[:,:] = 0.
            
    for i in range(pad_out.shape[1]):
        ph_pad = np.roll(th_pad, i, axis=1)
        pad_out += ph_pad
            
    return pad_out / pad_out.shape[1]
#==================================================
#==================================================
cpdef np.ndarray average_pad_rf(np.ndarray coeffs,
                np.ndarray LlpMmp, 
                np.ndarray weights,
                int n_th, int n_ph):
    
    cdef int i, j
    cdef np.ndarray bet, sinbe
    cdef np.ndarray angs_c, pad_out, angs_cr, thth_r, phph_r, 
    cdef np.ndarray th_pad, ph_pad, pad_r
    
    bet = np.linspace(0, np.pi, weights.shape[0])
    sinbe = np.sin(bet)
    angs, unit = angle_map(n_th, n_ph), np.ones((n_th,n_ph))
    angs_c = map_cart(unit, angs[0], angs[1])
    pad_out=np.zeros((n_th, n_ph), dtype=np.float_)
    for j in range(len(bet)):
        angs_cr = eulery(angs_c, bet[j])
        thth_r, phph_r = map_ang(angs_cr)[:2]
        pad_r = eval_pad_real(coeffs, LlpMmp, thth_r, phph_r, 
                              bet[j], 0.)
        pad_out += pad_r *sinbe[j] * weights[j]
    th_pad = pad_out / (n_th)
    pad_out[:,:] = 0.
            
    for i in range(pad_out.shape[1]):
        ph_pad = np.roll(th_pad, i, axis=1)
        pad_out += ph_pad
            
    return pad_out / pad_out.shape[1]
