import sys,os,re,glob,math,glob,signal,numpy
import iulib,ocropus
from pylab import *
from scipy.ndimage import filters

class Record:
    def __init__(self,**kw):
        self.__dict__.update(kw)
    def like(self,obj):
        self.__dict__.update(obj.__dict__)
        return self

def pad_to(image,w,h):
    """Symmetrically pad the image to the given width and height."""
    iw,ih = image.shape
    wd = int(w-iw)
    assert wd>=0
    w0 = wd/2
    w1 = wd-w0
    hd = int(h-ih)
    assert hd>=0
    h0 = hd/2
    h1 = hd-h0
    result = zeros((w,h))
    result[w0:w0+iw,h0:h0+ih] = image
    return result

def pad_by(image,r):
    """Symmetrically pad the image by the given amount"""
    w,h = image.shape
    result = zeros((w+2*r,h+2*r))
    result[r:(w+r),r:(h+r)] = image
    return result

def pad_bin(char,r=10):
    """Pad to the next bin size."""
    w,h = char.shape
    w = r*int((w+r-1)/r)
    h = r*int((h+r-1)/r)
    return pad_to(char,w,h)

def number_of_processors():
    try:
        return int(os.popen("cat /proc/cpuinfo  | grep 'processor.*:' | wc -l").read())
    except:
        return 1

def simple_classify(model,inputs):
    result = []
    for i in range(len(inputs)):
        result.append(model.coutputs(inputs[i]))
    return result

def omp_classify(model,inputs):
    if not "ocropus." in str(type(model)):
        return simple_classify(model,inputs)
    omp = ocropus.make_OmpClassifier()
    omp.setClassifier(model)
    n = len(inputs)
    omp.resize(n)
    for i in range(n):
        omp.input(inputs[i],i)
    omp.classify()
    result = []
    for i in range(n):
        outputs = ocropus.OutputVector()
        omp.output(outputs,i)
        outputs = model.outputs2coutputs(outputs)
        result.append(outputs)
    return result

def findfile(name):
    """Find some OCRopus-related resource by looking in a bunch off standard places.
    (FIXME: The implementation is pretty adhoc for now.
    This needs to be integrated better with setup.py and the build system.)"""
    local = "/usr/local/share/ocropus/"
    path = name
    if os.path.exists(path) and os.path.isfile(path): return path
    path = local+name
    if os.path.exists(path) and os.path.isfile(path): return path
    path = local+"/gui/"+name
    if os.path.exists(path) and os.path.isfile(path): return path
    path = local+"/models/"+name
    if os.path.exists(path) and os.path.isfile(path): return path
    path = local+"/words/"+name
    if os.path.exists(path) and os.path.isfile(path): return path
    _,tail = os.path.split(name)
    path = tail
    if os.path.exists(path) and os.path.isfile(path): return path
    path = local+tail
    if os.path.exists(path) and os.path.isfile(path): return path
    raise IOError("file '"+path+"' not found in . or /usr/local/share/ocropus/")

def finddir(name):
    """Find some OCRopus-related resource by looking in a bunch off standard places.
    (This needs to be integrated better with setup.py and the build system.)"""
    local = "/usr/local/share/ocropus/"
    path = name
    if os.path.exists(path) and os.path.isdir(path): return path
    path = local+name
    if os.path.exists(path) and os.path.isdir(path): return path
    _,tail = os.path.split(name)
    path = tail
    if os.path.exists(path) and os.path.isdir(path): return path
    path = local+tail
    if os.path.exists(path) and os.path.isdir(path): return path
    raise IOError("file '"+path+"' not found in . or /usr/local/share/ocropus/")

def imsdl(image,wait=1,norm=1):
    """Display the given iulib image using iulib's own
    display code."""
    iulib.dinit(512,512,1)
    flag = iulib.dactivate(1)
    if norm:
        iulib.dshown(image)
    else:
        iulib.dshow(image)
    if wait: 
        print "(click to continue)"
        iulib.dwait()
    iulib.dactivate(flag)

def allsplitext(path):
    """Split all the pathname extensions, so that "a/b.c.d" -> "a/b", ".c.d" """
    match = re.search(r'((.*/)*[^.]*)([^/]*)',path)
    if not match:
        return path,""
    else:
        return match.group(1),match.group(3)

def write_text(file,s):
    """Write the given string s to the output file."""
    with open(file,"w") as stream:
        if type(s)==unicode: s = s.encode("utf-8")
        stream.write(s)

def plotgrid(data,d=10,shape=(30,30)):
    ion()
    gray()
    clf()
    for i in range(min(d*d,len(data))):
        subplot(d,d,i+1)
        row = data[i]
        if shape is not None: row = row.reshape(shape)
        imshow(row)
    ginput(1,timeout=1)

def chist(l):
    counts = {}
    for c in l:
        counts[c] = counts.get(c,0)+1
    hist = [(v,k) for k,v in counts.items()]
    return sorted(hist,reverse=1)

################################################################
### simple database utilities        
################################################################

import sqlite3

def image2blob(image):
    # print image.shape,image.dtype
    if image.dtype!=numpy.dtype('B'):
        image = image*255.0+0.5
        image = numpy.array(image,'B')
    assert image.dtype==numpy.dtype('B'),image.dtype
    d0,d1 = image.shape
    assert d0>=0 and d0<256
    assert d1>=0 and d1<256
    s = numpy.zeros(d0*d1+2,'B')
    s[0] = d0
    s[1] = d1
    s[2:] = image.flat
    return buffer(s)

def blob2image(s):
    d0 = ord(s[0])
    d1 = ord(s[1])
    assert len(s)==d0*d1+2,(len(s),d0,d1)
    return numpy.frombuffer(s[2:],dtype='B').reshape(d0,d1)

class DbRow(sqlite3.Row):
    def __getattr__(self,name):
        return self[name]

def dbinsert(db,table,**assignments):
    cols = ""
    vals = ""
    values = []
    for k,v in assignments.items():
        if cols!="": cols += ","
        cols += k
        if vals!="": vals += ","
        vals += "?"
        values.append(v)
    cmd = "insert or replace into "+table+" ( "+cols+" ) values ( "+vals+" ) "
    params = list(values)
    # print cmd,params
    cur = db.cursor()
    cur.execute(cmd,params)
    cur.close()
    del cur

def dbcolumns(con,table,**kw):
    """Ensures that the table exists and that the given columns exist
    in the table; adds columns as necessary.  Columns are specified as
    colname="type" in the argument list."""
    cur = con.cursor()
    cols = list(cur.execute("pragma table_info("+table+")"))
    colnames = [col[1] for col in cols]
    if colnames==[]:
        cmd = "create table "+table+" (id integer primary key"
        for k,v in kw.items():
            cmd += ", %s %s"%(k,v)
        cmd += ")"
        cur.execute(cmd)
    else:
        # table already exists; add any missing columns
        for k,v in kw.items():
            if not k in colnames:
                cmd = "alter table "+table+" add column "+k+" "+v
                cur.execute(cmd)
    con.commit()
    del cur

def charcolumns(con,table):
    """Set up the columns for a character or cluster table.  It is safe to call
    this on existing tables; it will simply add any missing columns and indexes."""
    dbcolumns(con,table,
              # basic classification
              image="blob",
              cls="text",
              cost="real",
              # separate prediction
              # pocost is the cost for the transcribed cls
              pred="text",
              pcost="real",
              pocost="real",
              # cluster information
              cluster="integer",
              count="integer",
              classes="text",
              # geometry
              rel="text",
              lgeo="text",
              # line source
              file="text",
              segid="integer",
              bbox="text",
              )
    con.execute("create index if not exists cls_index on %s (cls)"%table)
    con.execute("create index if not exists cluster_index on %s (cluster)"%table)
    con.execute("create index if not exists cost_index on %s (cost)"%table)
    con.execute("create index if not exists countcost_index on %s (count,cost)"%table)
    con.commit()
              
def chardb(fname,table=None):
    db = sqlite3.connect(fname)
    db.row_factory = DbRow
    db.text_factory = sqlite3.OptimizedUnicode
    if table is not None:
        charcolumns(db,table)
        db.commit()
    db.execute("pragma synchronous=0")
    return db

