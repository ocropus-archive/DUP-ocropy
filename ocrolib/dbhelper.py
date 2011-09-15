################################################################
### simple database utilities        
################################################################

import sys,os,re,glob,math,glob,signal,numpy
from pylab import *
from scipy.ndimage import filters

import sqlite3

def image2blob(image):
    """Convert a numpy image to a blob that can be stored in a database.
    This only works for up to 255x255 byte images."""
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
    """Convert a database blob back to an image.  This only works
    for up to 255x255 byte images."""
    d0 = ord(s[0])
    d1 = ord(s[1])
    assert len(s)==d0*d1+2,(len(s),d0,d1)
    return numpy.frombuffer(s[2:],dtype='B').reshape(d0,d1)

class DbRow(sqlite3.Row):
    """A DB row that allows columns to be accessed as attributes."""
    def __getattr__(self,name):
        return self[name]

def dbinsert(db,table,**assignments):
    """Insert a database row with the columns and values as keyword
    arguments."""
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
    """Create a database for the given file name with the columns set up
    for a character database."""
    db = sqlite3.connect(fname)
    db.row_factory = DbRow
    db.text_factory = sqlite3.OptimizedUnicode
    if table is not None:
        charcolumns(db,table)
        db.commit()
    db.execute("pragma synchronous=0")
    return db

def ids(db,table):
    cur = db.cursor()
    return [x[0] for x in cur.execute("select id from %s"%table)]
