################################################################
### various document image processing utilities
################################################################

import scipy
from scipy import stats
from scipy.ndimage import measurements
from pylab import *

from common import *

def avg(*args):
    return mean(args)

def seg_boxes(seg,math=0):
    """Given a color segmentation, return a list of bounding boxes.
    Bounding boxes are returned as tuples (y0,y1,x0,x1).  With
    math=0, raster coordinates are used, with math=1, Postscript
    coordinates are used (however, the order of the values in the
    tuple doesn't change)."""
    seg = array(seg,'uint32')
    slices = measurements.find_objects(seg)
    h = seg.shape[0]
    result = []
    for i in range(len(slices)):
        if slices[i] is None: continue
        (ys,xs) = slices[i]
        if math:
            result += [(h-ys.stop-1,h-ys.start-1,xs.start,xs.stop)]
        else:
            result += [(ys.start,ys.stop,xs.start,xs.stop)]
    return result

seg_geometry_display = 0

def seg_geometry(segmentation,math=1):
    """Given a line segmentation (either an rseg--preferably connected
    component based--or a cseg, return (mh,a,b), where mh is the
    medium component height, and y=a*x+b is a line equation (in
    Postscript coordinates) for the center of the text line.  This
    function is used as a simple, standard estimator of text line
    geometry.  The intended use is to encode the size and centers of
    bounding boxes relative to these estimates and add these as
    features to the input of a character classifier, allowing it to
    distinguish otherwise ambiguous pairs like ,/' and o/O."""
    boxes = seg_boxes(segmentation,math=math)
    heights = [(y1-y0) for (y0,y1,x0,x1) in boxes]
    mh = stats.scoreatpercentile(heights,per=40)
    centers = [(avg(y0,y1),avg(x0,x1)) for (y0,y1,x0,x1) in boxes]
    xs = array([x for y,x in centers])
    ys = array([y for y,x in centers])
    if seg_geometry_display:
        if not math: plot(xs,ys,"g.")
        else: plot(xs,segmentation.shape[0]-ys-1,"g.")
    a,b = polyfit(xs,ys,1)
    return mh,a,b

def normalize_line_image(line,geo=None,target_h=32,target_mh=16):
    if geo is None:
        seg = segment_line(line)
        geo = seg_geometry(seg)
    # put ax+b in the center, then rescale so that mh is half height or so
    raise Error("unimplemented")

def rel_char_geom(box,params):
    """Given a character bounding box and a set of line geometry parameters,
    compute relative character position and size."""
    y0,y1,x0,x1 = box
    assert y1>y0 and x1>x0,"%s %s"%((x0,x1),(y0,y1))
    mh,a,b = params
    y = avg(y0,y1)
    x = avg(x0,x1)
    yl = a*x+b
    rel_ypos = (y-yl)/mh
    rel_width = (x1-x0)*1.0/mh
    rel_height = (y1-y0)*1.0/mh
    # ensure some reasonable bounds
    assert rel_ypos>-100 and rel_ypos<100
    assert rel_width>0 and rel_width<100
    assert rel_height>0 and rel_height<100
    return rel_ypos,rel_width,rel_height

def rel_geo_normalize(rel):
    """Given a set of geometric parameters, normalize them into the
    range -1...1 so that they can be used as input to a neural network."""
    if rel is None: return None
    if type(rel)==str:
        rel = [float(x) for x in rel.split()]
    ry,rw,rh = rel
    if not (rw>0 and rh>0): return None
    ry = clip(2*ry,-1.0,1.0)
    rw = clip(log(rw),-1.0,1.0)
    rh = clip(log(rh),-1.0,1.0)
    geometry = array([ry,rw,rh],'f')
    return geometry

def bbox(image):
    """Compute the bounding box for the pixels in the image."""
    assert len(image.shape)==2,"wrong shape: "+str(image.shape)
    image = array(image!=0,'uint32')
    cs = scipy.ndimage.measurements.find_objects(image)
    if len(cs)<1: return None
    c = cs[0]
    return (c[0].start,c[1].start,c[0].stop,c[1].stop)

def extract(image,bbox):
    """Extract a subregion of the given image.  The limits do not have to
    be within the image."""
    r0,c0,r1,c1 = bbox
    assert r0<=r1 and c0<=c1,"%s"%(bbox,)
    result = scipy.ndimage.interpolation.affine_transform(image,diag([1,1]),
                                                          offset=(r0,c0),
                                                          output_shape=(r1-r0,c1-c0))
    assert result.shape == (r1-r0,c1-c0),"docproc.extract failed: %s != %s"%(result.shape,(r1-r0,c1-c0))
    return result

def isotropic_rescale(image,r=32):
    """Rescale the image such that the non-zero pixels fall within a box of size
    r x r.  Rescaling is isotropic."""
    x0,y0,x1,y1 = bbox(image)
    sx = r*1.0/(x1-x0)
    sy = r*1.0/(y1-y0)
    s = min(sx,sy)
    s = min(s,1.0)
    rs = r/s
    dx = x0-(rs-(x1-x0))/2
    dy = y0-(rs-(y1-y0))/2
    result = scipy.ndimage.affine_transform(image,
                                            diag([1/s,1/s]),
                                            offset=(dx,dy),
                                            order=0,
                                            output_shape=(r,r))
    return result
