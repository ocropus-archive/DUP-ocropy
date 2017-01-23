################################################################
### text image generation with Cairo
################################################################

from __future__ import print_function

import ctypes
import cairo
from cairoextras import *
from numpy import *
from scipy import *
from scipy.misc import imsave
from pylab import *
import pango,pangocairo
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt,binary_erosion,binary_dilation
from scipy.ndimage.interpolation import map_coordinates,zoom,rotate
from scipy.stats.mstats import mquantiles


facecache = {}

def cairo_render_string(s,fontname=None,fontfile=None,size=None,bg=(0.0,0.0,0.0),fg=(0.9,0.9,0.9),pad=5):
    """Render a string using Cairo and the Cairo text rendering interface.  Fonts can either be given
    as a fontfile or as a fontname.  Size should be in pixels (?).  You can specify a background and
    foreground color as RGB floating point triples.  Images are padded by pad pixels on all sides."""
    face = None
    if fontfile is not None:
        # "/usr/share/fonts/truetype/msttcorefonts/comic.ttf"
        if fontfile in facecache:
            face = facecache[fontfile]
        else:
            face = create_cairo_font_face_for_file(fontfile,0)
            facecache[fontfile] = face

    # make a guess at the size
    w = max(100,int(size*len(s)))
    h = max(100,int(size*1.5))
    # possibly run through twice to make sure we get the right size buffer
    for round in range(2):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,w,h)
        cr = cairo.Context(surface)
        if face is not None:
            cr.set_font_face(face)
        else:
            if fontname is None: fontname = "Helvetica"
            cr.select_font_face(fontname)
        if size is not None:
            cr.set_font_size(size)
        xbear,ybear,tw,th = cr.text_extents(s)[:4]
        tw += 2*pad
        th += 2*pad
        if tw<=w and th<=h: break
        w = tw
        h = th
        
    cr.set_source_rgb(*bg)
    cr.rectangle(0,0,w,h)
    cr.fill()
    cr.move_to(-xbear+pad,-ybear+pad)
    cr.set_source_rgb(*fg)
    cr.show_text(s)

    data = surface.get_data()
    data = bytearray(data)
    a = array(data,'B')
    a.shape = (h,w,4)
    a = a[:th,:tw,:3]
    a = a[:,:,::-1]
    return a

def cairo_render_gray(*args,**kw):
    return mean(cairo_render_string(*args,**kw),axis=2)

def pango_families():
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,1,1)
        cr = cairo.Context(surface)
        pcr = pangocairo.CairoContext(cr)
        layout = pcr.create_layout()
        pcx = layout.get_context()
        return [f.get_name() for f in pcx.list_families()]
    
def pango_render_string(s,spec=None,fontfile=None,size=None,bg=(0.0,0.0,0.0),fg=(0.9,0.9,0.9),pad=5,
                        markup=1,scale=2.0,aspect=1.0,rotation=0.0):
    """Render a string using Cairo and the Pango text rendering interface.  Fonts can either be given
    as a fontfile or as a fontname.  Size should be in pixels (?).  You can specify a background and
    foreground color as RGB floating point triples. (Currently unimplemented.)"""
    S = pango.SCALE
    face = None
    if fontfile is not None: raise Exception("can't load ttf file into Pango yet; use fontname")
    # make a guess at the size
    w = max(100,int(scale*size*len(s)))
    h = max(100,int(scale*size*1.5))
    # possibly run through twice to make sure we get the right size buffer
    for round in range(2):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,w,h)
        cr = cairo.Context(surface)
        if spec is not None: fd = pango.FontDescription(spec)
        else: fd = pango.FontDescription()
        if size is not None: fd.set_size(int(scale*size*S))
        pcr = pangocairo.CairoContext(cr)
        layout = pcr.create_layout()
        layout.set_font_description(fd)
        if not markup:
            layout.set_text(s)
        else:
            layout.set_markup(s)
        ((xbear,ybear,tw,th),_) = layout.get_pixel_extents()
        # print(xbear, ybear, tw, th)
        tw = tw+2*pad
        th = th+2*pad
        if tw<=w and th<=h: break
        w = tw
        h = th
        
    cr.set_source_rgb(*bg)
    cr.rectangle(0,0,w,h)
    cr.fill()
    cr.move_to(-xbear+pad,-ybear+pad)
    cr.set_source_rgb(*fg)
    pcr.show_layout(layout)

    data = surface.get_data()
    data = bytearray(data)
    a = array(data,'B')
    a.shape = (h,w,4)
    a = a[:th,:tw,:3]
    a = a[:,:,::-1]
    if rotation!=0.0: a = rotate(a,rotation,order=1)
    a = zoom(a,(aspect/scale,1.0/scale/aspect,1.0),order=1)
    return a

def pango_render_gray(*args,**kw):
    return mean(pango_render_string(*args,**kw),axis=2)

def gauss_degrade(image,margin=1.0,change=None,noise=0.02,minmargin=0.5,inner=1.0):
    if image.ndim==3: image = mean(image,axis=2)
    m = mean([amin(image),amax(image)])
    image = 1*(image>m)
    if margin<minmargin: return 1.0*image
    pixels = sum(image)
    if change is not None:
        npixels = int((1.0+change)*pixels)
    else:
        edt = distance_transform_edt(image==0)
        npixels = sum(edt<=(margin+1e-4))
    r = int(max(1,2*margin+0.5))
    ri = int(margin+0.5-inner)
    if ri<=0: mask = binary_dilation(image,iterations=r)-image
    else: mask = binary_dilation(image,iterations=r)-binary_erosion(image,iterations=ri)
    image += mask*randn(*image.shape)*noise*min(1.0,margin**2)
    smoothed = gaussian_filter(1.0*image,margin)
    frac = max(0.0,min(1.0,npixels*1.0/prod(image.shape)))
    threshold = mquantiles(smoothed,prob=[1.0-frac])[0]
    result = (smoothed>threshold)
    return 1.0*result

def gauss_distort(images,maxdelta=2.0,sigma=10.0):
    n,m = images[0].shape
    deltas = randn(2,n,m)
    deltas = gaussian_filter(deltas,(0,sigma,sigma))
    deltas /= max(amax(deltas),-amin(deltas))
    deltas *= maxdelta
    xy = transpose(array(meshgrid(range(n),range(m))),axes=[0,2,1])
    # print(xy.shape, deltas.shape)
    deltas +=  xy
    return [map_coordinates(image,deltas,order=1) for image in images]

if __name__=="__main__":
    # print(sorted(pango_families()))
    ion()
    show()
    while 1:
        image = pango_render_string("A",spec="Arial",size=24,pad=20,scale=4.0)
        image = average(image,axis=2)
        for i in range(7):
            for j in range(7):
                noise = gauss_degrade(image,margin=(i-2)*0.5,noise=j*0.2)
                noise = gauss_distort([noise],maxdelta=1.0)[0]
                gray()
                subplot(7,7,7*i+j+1); imshow(noise)
                draw()
        raw_input()

def cairo_render_at(s,loc=None,shape=None,
                    fontname=None,fontfile=None,size=None,
                    slant=cairo.FONT_SLANT_NORMAL,
                    weight=cairo.FONT_WEIGHT_NORMAL,
                    bg=(0.0,0.0,0.0),fg=(0.9,0.9,0.9)):
    """Render a string using Cairo and the Cairo text rendering interface.  Fonts can either be given
    as a fontfile or as a fontname.  Size should be in pixels (?).  You can specify a background and
    foreground color as RGB floating point triples.  Images are padded by pad pixels on all sides."""
    assert loc is not None
    assert shape is not None
    assert size is not None
    w,h = shape
    x,y = loc
    face = None
    if fontfile is not None:
        # "/usr/share/fonts/truetype/msttcorefonts/comic.ttf"
        if fontfile in facecache:
            face = facecache[fontfile]
        else:
            face = create_cairo_font_face_for_file(fontfile,0)
            facecache[fontfile] = face

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,w,h)
    cr = cairo.Context(surface)
    if face is not None:
        cr.set_font_face(face)
    else:
        if fontname is None: fontname = "Helvetica"
        if type(slant)==str:
            if slant[0]=="i": slant = cairo.FONT_SLANT_ITALIC
            elif slant[0]=="o": slant = cairo.FONT_SLANT_OBLIQUE
            elif slant[0]=="n": slant = cairo.FONT_SLANT_NORMAL
            else: raise Exception("bad font slant specification (use n/i/o)")
        if type(weight)==str:
            if weight[0]=="b": weight = cairo.FONT_WEIGHT_BOLD
            elif weight[0]=="n": weight = cairo.FONT_WEIGHT_NORMAL
            else: raise Exception("bad font weight specification (use b/n)")
        cr.select_font_face(fontname,slant,weight)
    if size is not None:
        cr.set_font_size(size)
    cr.set_source_rgb(*bg)
    cr.rectangle(0,0,w,h)
    cr.fill()
    cr.move_to(x,y)
    cr.set_source_rgb(*fg)
    cr.show_text(s)
    data = surface.get_data()
    data = bytearray(data)
    a = array(data,'B')
    a.shape = (h,w,4)
    a = a[:,:,:3]
    a = a[:,:,::-1]
    return a

if __name__=="x__main__":
    s = u"hello, world: \u00E4\u0182\u03c0\u4eb0"
    subplot(311)
    imshow(cairo_render_string(s,fontname="Georgia",size=99,fg=(0.9,0.7,0.1),bg=(0.0,0.0,0.5)))
    subplot(312)
    font = "/usr/share/fonts/truetype/ttf-sil-gentium/GenR102.ttf"
    imshow(cairo_render_string(s,fontfile=font,size=70))
    subplot(313)
    font = "/usr/share/fonts/truetype/takao/TakaoPGothic.ttf"
    imshow(cairo_render_string(s,fontfile=font,size=70))
    show()

