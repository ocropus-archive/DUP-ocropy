import sys,os,re,glob,math,glob,signal
import iulib,ocropus


# TODO:
# -- buffer operations
# -- add image processing operations

class ArrayMixins:
    # other operations:
    
    # length()
    # rank()
    # dim(d)
    # clear()
    # resize(...)
    # reshape(...)
    # at(...)
    # put(...)
    # at1d(index)
    # put1d(index)
    # set(value,value,...)
    # fill(value)
    # copy(other)
    # append(other)
    # swap(other)
    # move(other)
    # copy(other)

    # common utilities
    def dims(self):
        """Like shape, but a method not a field."""
        result = []
        for i in range(4):
            d = self.dim(i)
            if d==0: break
            result.append(d)
        return result
    def duplicate(self):
        """Duplicate the current array"""
        result = self.new(*self.dims())
        result.copy(self)
        return result
    def makelike(self,other):
        """Changes the size and shape of this array
        to be like the other array."""
        iulib.makelike(self,other)
    def refcount(self):
        """Returns the Python reference count of this array."""
        return iulib.refcount(self)
    def init(self,l):
        return self.F(array(l))
    def __str__(self):
        l = []
        for i in range(min(5,self.length())):
            l.append(self.at(i))
        suffix = ""
        if self.length()>5: suffix = "..."
        result = "narray(%s %s%s)"%(self.dims(),l,suffix)
    def prt(self):
        print self.N()
    
    # conversions
    def numpy(self):
        """Convert an narray to a numpy array."""
        return iulib.numpy(self)
    def byte(self):
        """Return a bytearray version of the current array.
        May either share or copy."""
        result = bytearray()
        result.copy(self)
        return result
    def int(self):
        """Return an intarray version of the current array.
        May either share or copy."""
        result = intarray()
        result.copy(self)
        return result
    def float(self):
        """Return an floatarray version of the current array.
        May either share or copy."""
        result = floatarray()
        result.copy(self)
        return result

    def N(self):
        """Convert an narray to a numpy array."""
        return iulib.numpy(self)
    def NI(self):
        """Convert an narray to a numpy array, accounting for
        different coordinate conventions."""
        return transpose(N(self))[::-1,...]
    def F(self):
        """Assign a numpy array to this narray; returns self.
        Use like floatarray().F(numpy_array)."""
        a = iulib.floatarray()
        iulib.narray_of_numpy(self,image)
        return self
    def FI(self):
        """Assign a numpy array to this narray; returns self.
        Accounts for different coordinate conventions.
        Use like floatarray().FI(numpy_array)."""
        iulib.narray_of_numpy(self,transpose(image[::-1,...]))
        return self

    # misc operators
    def __getitem__(self,index):
        """Subscripting.  Can be used to retrieve elements.
        Also supports some slicing (currently, only a[u:v,s:t] is supported)"""
        if len(index)==2 and type(index[0])==slice and type(index[1])==slice:
            x,y = (index[0],index[1])
            assert x.step is None or x.step==1
            assert x.step is None or x.step==1
            return self.extract(x.start,y.start,x.stop,y.stop)
        else:
            return self.at(*index)
    def __setitem__(self,index,value):
        """Subscripting.  Can be used to retrieve elements.
        Also supports some slicing
        (currently, only a[u:v,s:t] = array and a[u:v,s:t] = value are supported)"""
        if len(index)==2 and type(index[0])==slice and type(index[1])==slice:
            x,y = (index[0],index[1])
            assert x.step is None or x.step==1
            assert x.step is None or x.step==1
            if type(value) is int or type(value) is float:
                self.fillRectangle(x.start,y.start,x.stop,y.stop,value)
            else:
                self.copyRectangle(x.start,y.start,value,0,0,x.stop-x.start,y.stop-y.start)
        else:
            self.put(list(index)+[value])
    def extract(self,x0,y0,x1,y1):
        """Extract a subrectangle from an image."""
        result = self.new()
        iulib.extract_subimage(result,self,x0,y0,x1,y1)
        return result
    def bextract(self,x0,y0,x1,y1,value):
        """Extract a subractangle, filling in values outside the boundary."""
        result = self.new()
        iulib.extract_bat(result,self,x0,y0,x1,y1,value)
        return result
    def indexOf(self,elt):
        """Find the index of the given element."""
        raise Unimplemented()
    def contains(self,elt):
        """Check whether the array contains the given element."""
        raise Unimplemented()
    def countOf(self,elt):
        """Number of times the element occurs in the given array."""
        raise Unimplemented()
    def ext(self,*args):
        """Returns a pixel, extending at the boundary."""
        return iulib.ext(self,*args)
    def bat(self,*args):
        """Returns a pixel, picking a fixed value outside the boundary."""
        return iulib.bat(self,*args)
    def bilin(self,x,y):
        """Returns a pixel with bilinear interpolation."""
        return iulib.bilin(self,x,y)
    def xref(self,x,y):
        return iulib.xref(self,x,y)


    # assignment versions
    def __iadd__(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.add(self,other)
    def __isub__(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.sub(self,other)
    def __imul__(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.mul(self,other)
    def __idiv__(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.div(self,other)
    def __itruediv__(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.div(self,other)
    def __ipow__(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.pow(self,other)
    def __imod__(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.div(self,other)
    def __ifloordiv__(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.div(self,other)
    def __ilshift(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.bit_lshift(self,other)
    def __irshift(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.bit_rshift(self,other)
    def __iand__(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.bit_and(self,other)
    def __ior__(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.bit_or(self,other)
    def __ixor__(self,other):
        """Assignment operator (+=, -=, ...)."""
        iulib.bit_xor(self,other)

    # infix versions implemented in terms of assignment
    def __add__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result += other
        return result
    def __sub__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result -= other
        return result
    def __mul__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result *= other
        return result
    def __div__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result /= other
        return result
    def __floordiv__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result //= other
        return result
    def __mod__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result %= other
        return result
    def __or__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result |= other
        return result
    def __and__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result &= other
        return result
    def __xor__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result ^= other
        return result
    def __power__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result **= other
        return result
    def __lshift__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result <<= other
        return result
    def __rshift__(self,other):
        """Infix operator (+, -, ...)."""
        result = self.duplicate()
        result >>= other
        return result

    # more arithmetic
    def abs(self):
        """Compute the absolute value in-place."""
        iulib.abs(self)
    def pos(self):
        """Compute +a in-place (no-op)."""
        pass
    def neg(self):
        """Compute -a in-place."""
        iulib.neg(self)
    def invert(self):
        """Invert the bits of a in-place."""
        iulib.invert(self)

    def __abs__(self):
        """Compute the absolute value of a and return it."""
        result = self.duplicate()
        iulib.abs(result)
        return result
    def __pos__(self):
        """Compute +a and return it."""
        return self.duplicate()
    def __neg__(self):
        """Compute -a and return it."""
        result = self.duplicate()
        iulib.neg(result)
        return result
    def __invert__(self):
        """Compute ~a and return it."""
        result = self.duplicate()
        iulib.invert(result)
        return result

    def log(self):
        """Compute log in-place."""
        iulib.log(self)
    def exp(self):
        """Compute exp in-place."""
        iulib.exp(self)
    def sin(self):
        """Compute sin in-place."""
        iulib.sin(self)
    def cos(self):
        """Compute cos in-place."""
        iulib.cos(self)

    def Log(self):
        """Compute log and return it."""
        result = self.duplicate()
        result.log()
        return result
    def Exp(self):
        """Compute exp and return it."""
        result = self.duplicate()
        result.exp()
        return result
    def Sin(self):
        """Compute sin and return it."""
        result = self.duplicate()
        result.sin()
        return result
    def Cos(self):
        """Compute cos and return it."""
        result = self.duplicate()
        result.cos()
        return result
    
    # min/max and related
    def greater(self,q,x,y):
        """Where self is bigger than q, assign x in-place, otherwise y."""
        iulib.greater(self,q,x,y)
    def less(self,q,x,y):
        """Where self is less than q, assign x in-place, otherwise y."""
        iulib.less(self,q,x,y)
    def min(self,other=None):
        """With no argument, returns the min of the array.
        With one argument, computes the min of this array
        with the other array in-place."""
        if other is None:
            return iulib.min(self)
        else:
            iulib.min(self,other)
    def max(self,other=None):
        """With no argument, returns the max of the array.
        With one argument, computes the max of this array
        with the other array in-place."""
        if other is None:
            return iulib.max(self)
        else:
            iulib.max(self,other)

    def Greater(self,q,x,y):
        """Where self is bigger than q, assign x, otherwise y;
        returns a new array."""
        result = self.duplicate()
        iulib.greater(result,q,x,y)
        return result
    def Less(self,q,x,y):
        """Where self is less than q, assign x, otherwise y;
        returns a new array."""
        result = self.duplicate()
        iulib.less(result,q,x,y)
        return result
    def Min(self,other=None):
        """With no argument, returns the min of the array.
        With one argument, computes the min of this array
        with the other array and returns it."""
        if other is None:
            return iulib.min(self)
        else:
            result = self.duplicate()
            iulib.min(result,other)
            return result
    def Max(self,other=None):
        """With no argument, returns the max of the array.
        With one argument, computes the max of this array
        with the other array and returns it."""
        if other is None:
            return iulib.max(self)
        else:
            result = self.duplicate()
            iulib.max(result,other)
            return result
        
    # more operations
    def sum(self):
        """Returns the sum of all the array elements."""
        return iulib.sum(self)
    def product(self):
        """Returns the product of all the array elements."""
        return iulib.product(self)
    def argmax(self):
        """Returns the 1D index of the largest element."""
        return iulib.argmax(self)
    def argmin(self):
        """Returns the 1D index of the smallest element."""
        return iulib.argmin(self)
    def norm2(self):
        """Returns the 2-norm of the array."""
        return iulib.norm2(self)
    def dist2(self,other):
        """Returns the Euclidean distance of the two arrays."""
        return iulib.dist2(self,other)

    # in-place modification
    def clamp(self,lo,hi):
        """Clamp the values of the array between lo and hi."""
        iulib.clamp(self,lo,hi)
    def clampScale(self,other,lo,hi):
        """Scale the array by multiplying with other, then clamp
        the result between lo and hi."""
        iulib.clampscale(self,other,lo,hi)
    def containsOnly(self,*args):
        """Check that the array contains only the given values
        (give one or two arguments)."""
        if len(args)==1:
            return iulib.contains_only(self,args[0])
        if len(args)==2:
            return iulib.contains_only(self,args[0],args[1])
        raise Unimplemented()
    def addScaled(self,other,x):
        """Add a scaled version of ther array to this array."""
        iulib.addScaled(self,other,x)
    def makeUnitVector(self,i,n):
        """Make this array a unit vector of length n, with a 1 in
        position i."""
        iulib.make_unit_vector(self,i,n)
    def reverse(self):
        """Reverse the elements of the array."""
        iulib.reverse(self)
    def removeLeft(self,x):
        iulib.removeLeft(self,x)
    def removeElement(self,i):
        """Remove element i"""
        iulib.removeElement(self,i)
    def removeValue(self,x):
        """Remove value x"""
        iulib.removeValue(self,x)
    def insertAt(self,i,v):
        """Insert value v at location i."""
        iulib.insert_at(self,i,v)
    def iota(self,n):
        """Assign the values of 0...n-1 to this array,
        then return the result."""
        iulib.iota(self,n)
        return self
    def randomly_permute(self):
        """Randomly permute the elements of this array."""
        iulib.randomly_permute(self)
    def normalize2(self):
        """Divide the elements of this array by the 2-norm."""
        iulib.normalize2(self)

    # image I/O
    def readGray(self,file):
        """Read an image as grayscale (always makes
        the array rank 2)."""
        iulib.read_image_gray(self,file)
    def readRgb(self,file):
        """Read an image as color (always makes
        the array rank 3)."""
        iulib.read_image_rgb(self,file)
    def readPacked(self,file):
        """Read an image as packed RGB (always makes
        the array rank 2)."""
        iulib.read_image_packed(self,file)
    def readBinary(self,file):
        """Read an image as binary (always makes
        the array rank 2)."""
        iulib.read_image_binary(self,file)
    def writeGray(self,file):
        """Write the array as a grayscale image."""
        iulib.write_image_gray(file,self)
    def writeRgb(self,file):
        """Write the array as an RGB image."""
        iulib.write_image_rgb(file,self)
    def writePacked(self,file):
        """Write the array as a packed RGB image."""
        iulib.write_image_packed(file,self)
    def writeBinary(self,file):
        """Write the array as a binary image."""
        iulib.write_image_binary(file,self)

    # more image operations
    def tighten(self):
        """Crop zero values around the outside of the array."""
        iulib.tighten(self)
    def cirularPermute(self,dx,dy):
        """Shift the values of the array by dx,dy, using
        circular boundary conditions."""
        iulib.circ_by(self,dx,dy)
    def shiftBy(self,dx,dy,value=0):
        """Shift the values of the array by dx,dy, using
        a default value at the boundary."""
        iulib.shift_by(self,dx,dy,value)
    def padBy(self,dx,dy,value=0):
        """Pad the array by the given amounts (positive or negative)."""
        iulib.pad_by(self,dx,dy,value)
    def eraseBoundary(self,dx,dy,value=0):
        """Fill values around the boundary with the given value."""
        iulib.erase_boundary(self,dx,dy,value)
    def resizeTo(self,w,h,value=0):
        """Resize the image to the given size, filling values around
        the boundary with the given default."""
        iulib.resize_to(self,w,h,value)
    def ifelse(iftrue,iffalse):
        """Use this array to select between the iftrue and iffalse
        arrays and return the result."""
        result = self.new()
        iulib.ifelse(result,self,iftrue,iffalse)
        return result
    def gammaTransform(self,gamma,c,lo,hi):
        """Perform a gamma transform."""
        iulib.gamma_transform(self,gamma,c,lo,hi)
    def expandRange(self,lo,hi):
        """Expand the range of the image to fall between lo and hi."""
        iulib.expand_range(self,lo,hi)
    def cropRectangle(self,*args):
        """Crop this image to the given rectangle."""
        temp = self.new()
        iulib.crop(temp,self,*args)
        self.move(temp)
    def crop(self,x0,y0,x1,y1):
        """Crop this image to the given rectangle."""
        temp = self.new()
        iulib.crop(temp,self,x0,y0,x1-x0,y1-y0)
        self.move(temp)
    def transpose(self):
        """Transpose this image."""
        iulib.tranpose(self)
    def replaceValues(old,new):
        """Replace the old value with the new value in this image."""
        iulib.replace_values(self,old,new)
    def threshold(self,t):
        """Threshold this image with the given threshold."""
        ocropus.binarize_with_threshold(self,t)
    def copyRectangle(x,y,src,x0,y0,x1,y1):
        """Copy the given source rectangle from the source image
        to this image at position x,y."""
        iulib.copy_rect(self,x,y,src,x0,y0,x1,y1)
    def rotate(angle,cx=None,cy=None,interpolate=1,direct=1):
        """Rotate this image using a variety of methods.

        cx,cy set the center of the rotation

        Angles of 0.0,90.0,180.0,270.0 with the default cx,cy

        use special, fast implementations.

        When interpolate=1 is set, uses bilinear interpolation,
        sampling otherwise.

        When direct=1 is set, uses direct rotation, skew-based
        rotation otherwise."""
        if cx is None and cy is None and float(angle) in [0.0,90.0,180.0,270.0]:
            if float(angle)==0.0:
                pass
            elif float(angle)==90.0:
                temp = self.new()
                rotate_90(temp,self)
                self.move(temp)
            elif float(angle)==180.0:
                temp = self.new()
                rotate_180(temp,self)
                self.move(temp)
            elif float(angle)==270:
                temp = self.new()
                rotate_270(temp,self)
                self.move(temp)
        if cx is None: cx = self.width()/2.0
        if cy is None: cy = self.height()/2.0
        if direct:
            if interpolate:
                temp = self.new()
                iulib.rotate_direct_interpolate(temp,self,angle,cx,cy)
                self.move(temp)
            else:
                temp = self.new()
                iulib.rotate_direct_sample(temp,self,angle,cx,cy)
                self.move(temp)
        else:
            raise Unimplemented()
    def scale(sx,sy=None,interpolate=1):
        """Scale this image using a variety of methods.

        When interpolate=1 is set, uses bilinear interpolation,
        sampling otherwise.

        When direct=1 is set, uses direct rotation, skew-based
        rotation otherwise."""
        if sy is None: sy = sx
        if sx==1.0 and sy==1.0: return
        if interpolate:
            temp = self.new()
            iulib.scale_interpolate(temp,self,float(sx),float(sy))
            self.move(temp)
        else:
            temp = self.new()
            iulib.scale_sample(temp,self,float(sx),float(sy))
            self.move(temp)
    def scaleTo(nx,ny=None,interpolate=1):
        """Scale this image to the given target size.

        When interpolate=1 is set, uses bilinear interpolation,
        sampling otherwise.

        When direct=1 is set, uses direct rotation, skew-based
        rotation otherwise."""
        if ny is None: ny = nx
        if interpolate:
            temp = self.new()
            iulib.scale_interpolate(temp,self,int(nx),int(ny))
            self.move(temp)
        else:
            temp = self.new()
            iulib.scale_sample(temp,self,int(nx),int(ny))
            self.move(temp)
    def splitRGB(self):
        """Split this image into r,g,b channels and return them."""
        assert self.rank()==3 and self.dim(2)==3
        r = self.new()
        g = self.new()
        b = self.new()
        iulib.split_rgb(r,g,b,self)
        return (r,g,b)
    def combineRGB(r,g,b):
        """Combine the three r,g,b channels into this image.
        Returns self."""
        iulib.combine_rgb(self,r,g,b)
        return self
    def math2raster(self):
        """Update this image from mathematical coordinates (a[x,y]) to
        raster coordinates (a[row,col])."""
        iulib.math2raster(self)
    def raster2math(self):
        """Update this image from raster coordinates (a[row,col]) to
        mathematical coordinates (a[x,y])."""
        iulib.raster2math(self)
    def fillRectangle(self,*args):
        """Fill the given rectangle with the given value.
        Invoke either as a.fillRecangle(r,value) or
        a.fillRectangle(x0,y0,x1,y1,value)."""
        iulib.fill_rect(self,*args)
    def gauss(self,s,sy=None):
        """Perform Gaussian convolution."""
        if self.rank()==1:
            iulib.gauss1d(self,s)
        elif self.rank()==2:
            if sy is None: sy = s
            iulib.gauss2d(self,s,sy)
        else:
            raise Unimplemented()
    def localMinima(self):
        """Compute a map of local minima and return them
        as a bytearray."""
        temp = bytearray()
        iulib.local_minima(temp,self)
        return temp
    def localMaxima(self):
        """Compute a map of local maxima and return them
        as a bytearray."""
        temp = bytearray()
        iulib.local_maxima(temp,self)
        return temp
    def zerocrossings(self):
        """Compute a map of zerocrossings and return them
        as a bytearray."""
        temp = bytearray()
        iulib.zero_crossings(temp,self)
        return temp
    def medianFilter(self,rx,ry):
        """Compute the median filter in-place."""
        iulib.median_filter(self,rx,ry)
    def corners(self,method="kr2"):
        """Find corners in this image."""
        result = floatarray()
        if method=="gradient":
            result.copy(self)
            iulib.gradient_based_corners(result)
        elif method=="kr":
            iulib.kitchen_rosenfeld_corners(result,self.float())
        elif method=="kr2":
            iulib.kitchen_rosenfeld_corners2(result,self.float())
        else:
            raise Unimplemented()
        return result
    # TODO
    # Canny edges
    # thinning
    # tracing

class intarray(iulib.intarray,ArrayMixins):
    def new(self,*args):
        return intarray(*args)
    def propagateLabels(self):
        iulib.propagateLabels(self)
    def propagateLabelsFrom(self,seed):
        iulib.propagate_labels_to(self,seed)
    def renumberLabels(self,start=1):
        iulib.renumber_labels(self,start)
    def labelComponents(self,four_connected=0):
        iulib.label_components(self,four_connected)
    def boundingBoxes(self):
        result = rectarray()
        iulib.bounding_boxes(result,self)
        return result
    def packRGB(self,r,g,b):
        iulib.pack_rgb(self,r,g,b)
    def unpackRGB(self):
        r = bytearray()
        g = bytearray()
        b = bytearray()
        iulib.unpack_rgb(r,g,b,self)
        return (r,g,b)
        
class bytearray(ArrayMixins,iulib.bytearray):
    def new(self,*args):
        return bytearray(*args)
    def boundingBoxes(self):
        """Compute bounding boxes for the non-zero
        elements of this array."""
        temp = intarray()
        temp.copy(self)
        return temp.boundingBoxes()
    def difference(self,other):
        """Compute the difference between this and the
        other array in-place."""
        iulib.difference(self,other)
    def Difference(self,other):
        """Compute the difference between this and the
        other array and return it."""
        iulib.difference(self,other)
        result = self.duplicate()
        iulib.difference(result,other)
        return result
    def maxdifference(self,other):
        """Compute the maximum absolute difference between
        this an the other array and return it."""
        return iulib.maxdifference(self,other)
    def shiftAnd(self,other,dx=0,dy=0):
        """Perform a morphological shifted 'and' operation in place."""
        iulib.binary_and(self,other,dx,dy)
    def shiftOr(self,other,dx=0,dy=0):
        """Perform a morphological shifted 'or' operation in place."""
        iulib.binary_or(self,other,dx,dy)
    def open(w=None,h=None,mask=None,radius=None,gray=0):
        """Perform morphological opening.
        If a mask is given, uses the mask.
        If a radius is given, uses a circular structuring element.
        Otherwise uses w/h and a rectangular structuring element.
        Performs grayscale operations if gray=1."""
        if gray:
            assert mask is not None
            iulib.gray_open(self,mask)
        else:
            if mask is not None:
                raise Unimplemented()
            elif radius is not None:
                iulib.binary_open_circle(self,radius)
            elif w is not None or h is not None:
                if w is None: w = 0
                if h is None: h = 0
                iulib.binary_open_rect(self,w,h)
    def close(w=None,h=None,mask=None,radius=None,gray=0):
        """Perform morphological closing.
        If a mask is given, uses the mask.
        If a radius is given, uses a circular structuring element.
        Otherwise uses w/h and a rectangular structuring element.
        Performs grayscale operations if gray=1."""
        if gray:
            assert mask is not None
            iulib.gray_close(self,mask)
        else:
            if mask is not None:
                raise Unimplemented()
            elif radius is not None:
                iulib.binary_close_circle(self,radius)
            elif w is not None or h is not None:
                if w is None: w = 0
                if h is None: h = 0
                iulib.binary_close_rect(self,w,h)
    def erode(w=None,h=None,mask=None,radius=None,gray=0):
        """Perform morphological erosion.
        If a mask is given, uses the mask.
        If a radius is given, uses a circular structuring element.
        Otherwise uses w/h and a rectangular structuring element.
        Performs grayscale operations if gray=1."""
        if gray:
            assert mask is not None
            iulib.gray_erode(self,mask)
        else:
            if mask is not None:
                raise Unimplemented()
            elif radius is not None:
                iulib.binary_erode_circle(self,radius)
            elif w is not None or h is not None:
                if w is None: w = 0
                if h is None: h = 0
                iulib.binary_erode_rect(self,w,h)
    def dilate(w=None,h=None,mask=None,radius=None,gray=0):
        """Perform morphological dilation.
        If a mask is given, uses the mask.
        If a radius is given, uses a circular structuring element.
        Otherwise uses w/h and a rectangular structuring element.
        Performs grayscale operations if gray=1."""
        if gray:
            assert mask is not None
            iulib.gray_dilate(self,mask)
        else:
            if mask is not None:
                raise Unimplemented()
            elif radius is not None:
                iulib.binary_dilate_circle(self,radius)
            elif w is not None or h is not None:
                if w is None: w = 0
                if h is None: h = 0
                iulib.binary_dilate_rect(self,w,h)

class floatarray(ArrayMixins,iulib.floatarray):
    def new(self,*args):
        return floatarray(*args)
        
class rectarray(ArrayMixins,iulib.rectarray):
    def new(self,*args):
        return floatarray(*args)
