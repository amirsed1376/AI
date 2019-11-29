import pylab as pyl
import math as m
vinit = 0.0
dt = 0.01
ena = 115
gna = 120
ek = -12
gk = 36
el = 10.6
gl = 0.3

def upd ( x , dltax ) :
    return ( x + dltax * dt )

def mnh0 (a,b):
    return (a/(a+b))


def am (v): return