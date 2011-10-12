################################################################
### Image I/O. This is in a separate module because it depends on the
### image I/O library that we are using.  This code is written using
### the iulib library, which works correctly with TIFF, JPG, and PNG.
### The built-in PIL library has some problems.
################################################################

import os
import iulib
from iulib import *
from iuutils import *

def iulib_page_iterator(files):
    """Given a list of files, iterate through the page images in those
    files.  When multi-page TIFF files are encountered, iterates through
    each such TIFF file.  Yields tuples (image,filename), where image
    is in iulib format."""
    for file in files:
        _,ext = os.path.splitext(file)
        if ext.lower()==".tif" or ext.lower()==".tiff":
            if os.path.getsize(file)>2e9:
                raise IOError("TIFF file is greater than 2G")
            tiff = iulib.Tiff(file,"r")
            for i in range(tiff.numPages()):
                image = iulib.bytearray()
                try:
                    tiff.getPageRaw(image,i,True)
                except:
                    tiff.getPage(image,i,True)
                yield image,"%s[%d]"%(file,i)
        else:
            image = iulib.bytearray()
            iulib.read_image_gray(image,file)
            yield image,file

def page_iterator(files):
    """Given a list of files, iterate through the page images in those
    files.  When multi-page TIFF files are encountered, iterates through
    each such TIFF file.  Yields tuples (image,filename), where image
    is in NumPy format."""
    # use the iulib implementation because its TIFF reader actually works;
    # the TIFF reader in all the Python libraries is broken
    for image,file in iulib_page_iterator(files):
        yield narray2page(image),file

def read_image_gray(file,type='B'):
    """Read an image in grayscale."""
    if not os.path.exists(file): raise IOError(file)
    na = iulib.bytearray()
    iulib.read_image_gray(na,file)
    return narray2numpy(na,type=type)

def write_image_gray(file,image):
    """Write an image in grayscale."""
    assert (array(image.shape)>0).all()
    if image.ndim==2:
        iulib.write_image_gray(file,numpy2narray(image))
    elif image.ndim==3:
        iulib.write_image_gray(file,numpy2narray(mean(image,axis=2)))
    else:
        raise Exception("ndim must be 2 or 3, not %d"%image.ndim)

def draw_pseg(pseg,axis=None):
    """Display a pseg."""
    if axis is None:
        axis = subplot(111)
    h = pseg.dim(1)
    regions = RegionExtractor()
    regions.setPageLines(pseg)
    for i in range(1,regions.length()):
        (r0,c0,r1,c1) = (regions.x0(i),regions.y0(i),regions.x1(i),regions.y1(i))
        p = patches.Rectangle((c0,r0),c1-c0,r1-r0,edgecolor="red",fill=0)
        axis.add_patch(p)

def write_page_segmentation(name,pseg,white=1):
    """Write a numpy page segmentation (rank 3, type='B' RGB image.)"""
    pseg = pseg2narray(pseg)
    if white: iulib.make_page_segmentation_white(pseg)
    iulib.write_image_packed(name,pseg)
    
def read_page_segmentation(name,black=1):
    """Write a numpy page segmentation (rank 3, type='B' RGB image.)"""
    if not os.path.exists(name): raise IOError(name)
    pseg = iulib.intarray()
    iulib.read_image_packed(pseg,name)
    if black: iulib.make_page_segmentation_black(pseg)
    return narray2pseg(pseg)
    
def write_line_segmentation(name,lseg,white=1):
    """Write a numpy line segmentation."""
    lseg = lseg2narray(lseg)
    if white: iulib.make_line_segmentation_white(lseg)
    iulib.write_image_packed(name,lseg)
    
def read_line_segmentation(name,black=1):
    """Write a numpy line segmentation."""
    if not os.path.exists(name): raise IOError(name)
    lseg = iulib.intarray()
    iulib.read_image_packed(lseg,name)
    if black: iulib.make_line_segmentation_black(lseg)
    return narray2lseg(lseg)

def renumber_labels(line,start):
    line = lseg2narray(line)
    iulib.renumber_labels(line,start)
    return narray2lseg(line)

