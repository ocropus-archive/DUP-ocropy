import sys,os,re,glob,math,glob,signal
import iulib,ocropus
from scipy.ndimage import interpolation

def square(image):
    """Make a numpy array square."""
    w,h = image.shape
    r = max(w,h)
    output = zeros((r,r),image.dtype)
    dx = (r-w)/2
    dy = (r-h)/2
    output[dx:dx+w,dy:dy+h] = image
    return output

def stdsize(image,r=30):
    """Make a numpy array a standard square size."""
    image = square(image)
    s,_ = image.shape
    return interpolation.zoom(image,(r+0.5)/float(s))

def center_maxsize(image,r):
    """Center the image and fit it into an r x r output image.
    If the input is larger in any dimension than r, it is
    scaled down."""
    from pylab import amin,amax,array,zeros
    assert amin(image)>=0 and amax(image)<=1
    image = array(image,'f')
    w,h = image.shape
    s = max(w,h)
    # zoom down, but don't zoom up
    if s>r:
        image = interpolation.zoom(image,(r+0.5)/float(s))
        image[image<0] = 0
        image[image>1] = 1
        w,h = image.shape
    output = zeros((r,r),image.dtype)
    dx = (r-w)/2
    dy = (r-h)/2
    output[dx:dx+w,dy:dy+h] = image
    return output

def blackout_images(image,ticlass):
    """Takes a page image and a ticlass text/image classification image and replaces
    all regions tagged as 'image' with rectangles in the page image.  The page image
    is modified in place.  All images are iulib arrays."""
    rgb = ocropy.intarray()
    ticlass.textImageProbabilities(rgb,image)
    r = ocropy.bytearray()
    g = ocropy.bytearray()
    b = ocropy.bytearray()
    ocropy.unpack_rgb(r,g,b,rgb)
    components = ocropy.intarray()
    components.copy(g)
    n = ocropy.label_components(components)
    print "[note] number of image regions",n
    tirects = ocropy.rectarray()
    ocropy.bounding_boxes(tirects,components)
    for i in range(1,tirects.length()):
        r = tirects.at(i)
        ocropy.fill_rect(image,r,0)
        r.pad_by(-5,-5)
        ocropy.fill_rect(image,r,255)
        
