#
# this is drity AF, it works
# Created by Taylor B. Sage
# TODO: Remove redundent datum 'stuff' pulled from: https://www.oc.nps.edu/oc2902w/coord/llhxyz.htm Javascript
# 

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# using wgs84 
# TODO: create a datum ojbect for this

wgs84a = 6378.137
wgs84f = 1.0/298.257223563
wgs84b = wgs84a * ( 1.0 - wgs84f )

a = wgs84a
b = wgs84b

f = 1-b/a
eccsq = 1 - b*b/(a*a)
ecc = np.sqrt(eccsq)

EARTH_A = a
EARTH_B = b
EARTH_F = f
EARTH_Ecc = ecc
EARTH_Esq = eccsq


def radcur(lati):

        dtr = np.pi/180.0

        a = EARTH_A
        b = EARTH_B

        asq   = a*a
        bsq   = b*b
        eccsq  =  1 - bsq/asq
        ecc = np.sqrt(eccsq)

        lat   =  lati

        clat  =  np.cos(dtr*lat)
        slat  =  np.sin(dtr*lat)

        dsq   =  1.0 - eccsq * slat * slat
        d     =  np.sqrt(dsq)

        rn    =  a/d
        rm    =  rn * (1.0 - eccsq ) / dsq

        rho   =  rn * clat
        z     =  (1.0 - eccsq ) * rn * slat
        rsq   =  rho*rho + z*z
        r     =  np.sqrt( rsq )

        return np.array([r, rn, rm])


def latlonelv_xyz(flat, flon, altkmi):
        dtr =  np.pi/180.0

        altkm = altkmi

        clat = np.cos(dtr*flat)
        slat = np.sin(dtr*flat)
        clon = np.cos(dtr*flon)
        slon = np.sin(dtr*flon)

        rrnrm  = radcur (flat)
        rn     = rrnrm[1]
        re     = rrnrm[0]

        ecc    = EARTH_Ecc
        esq    = ecc*ecc

        x =  (rn + altkm) * clat * clon
        y =  (rn + altkm) * clat * slon
        z =  ( (1-esq)*rn + altkm ) * slat

        return  np.array([x, y, z])


def rearth (lati):

    rrnrm =  radcur ( lati )
    r     =  rrnrm[0]
    return r


def gd2gc (flatgdi, altkmi ):
    dtr   = np.pi/180.0
    rtd   = 1/dtr

    flatgd = flatgdi
    altkm  =  altkmi

    ecc   =  EARTH_Ecc
    esq   =  ecc*ecc

    altnow  =  altkm

    rrnrm   =  radcur (flatgd)
    rn      =  rrnrm[1]

    ratio   = 1 - esq*rn/(rn+altnow)

    tlat    = pi.tan(dtr*flatgd) * ratio
    flatgc  = rtd * pi.arctan(tlat)

    return  flatgc


def gc2gd (flatgci, altkmi ):

    dtr   = np.pi/180.0
    rtd   = 1/dtr

    flatgc=  flatgci
    altkm =  altkmi

    ecc   =  EARTH_Ecc
    esq   =  ecc*ecc

    altnow  =  altkm

    rrnrm   =  radcur (flatgc)
    rn      =  rrnrm[1]

    ratio   = 1 - esq*rn/(rn+altnow)

    tlat    = np.tan(dtr*flatgc) / ratio
    flatgd  = rtd * np.arctan(tlat)

    rrnrm   =  radcur ( flatgd )
    rn      =  rrnrm[1]

    ratio   =  1  - esq*rn/(rn+altnow)
    tlat    =  np.tan(dtr*flatgc)/ratio
    flatgd  =  rtd * np.arctan(tlat)

    return  flatgd


def xyz_latlonelv ( xvec ):

    dtr =  np.pi/180.0
    esq    =  EARTH_Esq

    x      = xvec[0]
    y      = xvec[1]
    z      = xvec[2]

    rp = np.sqrt(xvec[0] * xvec[0] +  xvec[1] * xvec[1] +  xvec[2] * xvec[2] )
    
    flatgc = np.arcsin ( xvec[2] / rp )/dtr


    testval = np.abs(xvec[:1]).sum()

    if testval < 1.0e-10:
        flon = 0.0
    else:
        flon = np.arctan2 ( y,x )/dtr
    if flon < 0.0:
        flon = flon + 360.0

    p = np.linalg.norm(np.array([x,y]))
    if p < 1.0e-10:
        flat = 90.0
        if z < 0.0:
            flat = -90.0

        altkm = rp - rearth(flat)
        return  np.array([flat,flon,altkm])

    rnow  =  rearth(flatgc)
    altkm =  rp - rnow
    flat  =  gc2gd (flatgc,altkm)

    rrnrm =  radcur(flat)
    rn    =  rrnrm[1]

    for kount in range(5):
        slat  =  np.sin(dtr*flat)
        tangd =  ( z + rn*esq*slat ) / p
        flatn =  np.arctan(tangd)/dtr

        dlat  =  flatn - flat
        flat  =  flatn
        clat  =  np.cos( dtr*flat )

        rrnrm =  radcur(flat)
        rn    =  rrnrm[1]

        altkm =  (p/clat) - rn

        if np.abs(dlat) < 1.0e-12: 
            break
    
    return  np.array([flat,flon,altkm])


# Path prediction/interpolation 
# linear regression for a 3 degree poly based on a dataset of N length, projects ahead based on the time diff mean.

def interpolate(geoloc, time, elv):

    #Note: Lat long backwards (need to fix)
    pos_data = [[gt.latlonelv_xyz(geoloc[i,1], geloc[i,0], elv[i])] for i in range(geoloc.shape[0])]
    pos_data = np.vstack(pos_data)

    # setup the degree 3 poly
    poly = PolynomialFeatures(degree=3, include_bias=False)
    polyTransform = poly.fit_transform(time)

    # create and fit the Linear regression
    lr = LinearRegression()
    lr.fit(polyTransform,pos_data.reshape(-1,3))

    #calculate the mean time diff
    t = time[-1] + np.diff(time,axis=0).mean()
    p = lr.predict(poly.fit_transform(np.array([t])))

    return gt.xyz_latlonelv(p.reshape(-1))


