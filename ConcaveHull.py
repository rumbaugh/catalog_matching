import numpy as np
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import math
import matplotlib.pyplot as plt

#Adapted from http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/

def ConcaveHull(points,alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
 
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
 
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
 
    #coords = np.array([point.coords[0] for point in points])
    coords = points
 
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
 
        # Lengths of sides of triangle
        a = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = np.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = np.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
 
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
 
        # Area of triangle by Heron's formula
        area = np.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
 
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
 
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

def CHullRandomPoint(poly,npnts):
    ra_LB,dec_LB,ra_UB,dec_UB=poly.bounds
    rand_arr = np.random.random(2*npnts)
    ra_rand = (ra_UB-ra_LB)*rand_arr[:npnts] + ra_LB
    dec_rand = (dec_UB-dec_LB)*rand_arr[npnts:] + dec_LB
    testinpoly=np.zeros(npnts,dtype='bool')
    for iCR in range(0,npnts):
        testinpoly[iCR]=poly.contains(geometry.Point(ra_rand[iCR],dec_rand[iCR]))
    while np.sum(testinpoly)<npnts:
        repinds=True-testinpoly
        rand_arr = np.random.random(2*(npnts-np.sum(testinpoly)))
        ra_rand[repinds] = (ra_UB-ra_LB)*rand_arr[:npnts-np.sum(testinpoly)] + ra_LB
        dec_rand[repinds] = (dec_UB-dec_LB)*rand_arr[npnts-np.sum(testinpoly):] + dec_LB
        tmptest=testinpoly[repinds]
        for iCR in range(0,npnts-np.sum(testinpoly)):
            tmptest[iCR]=poly.contains(geometry.Point(ra_rand[repinds][iCR],dec_rand[repinds][iCR]))
        testinpoly[repinds]=tmptest
    return ra_rand,dec_rand

def CHullGaussRand(poly,npnts,sig,isradec=True):
    RP=poly.representative_point()
    pXcen,pYcen=RP.bounds[0],RP.bounds[1]
    if isradec:
        ra_rand=np.random.normal(pXcen,sig*np.cos(pYcen*np.pi/180),npnts)
    else:
        ra_rand=np.random.normal(pXcen,sig,npnts)
    dec_rand=np.random.normal(pYcen,sig,npnts)
    ra_LB,dec_LB,ra_UB,dec_UB=poly.bounds
    testinpoly=np.zeros(npnts,dtype='bool')
    for iCR in range(0,npnts):
        testinpoly[iCR]=poly.contains(geometry.Point(ra_rand[iCR],dec_rand[iCR]))
    while np.sum(testinpoly)<npnts:
        repinds=True-testinpoly
        newnum=npnts-np.sum(testinpoly)
        if isradec:
            ra_rand[repinds]=np.random.normal(pXcen,sig*np.cos(pYcen*np.pi/180),newnum)
        else:
            ra_rand[repinds]=np.random.normal(pXcen,sig,newnum)
            dec_rand[repinds]=np.random.normal(pYcen,sig,newnum)
        tmptest=testinpoly[repinds]
        for iCR in range(0,newnum-np.sum(testinpoly)):
            tmptest[iCR]=poly.contains(geometry.Point(ra_rand[repinds][iCR],dec_rand[repinds][iCR]))
        testinpoly[repinds]=tmptest
    return ra_rand,dec_rand

def CheckPoints(poly,ras,decs):
    g=[poly.contains(geometry.Point(ras[ip],decs[ip])) for ip in range(0,len(ras))]
    return np.arange(len(ras))[np.array(g)==True]
        
