import os
import sqlite3
import numpy
import docproc
import utils

verbose = os.getenv("dbutils_verbose")
if verbose!=None: verbose = int(verbose)

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

class CharRow(sqlite3.Row):
    def __getattr__(self,name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError()
    def byte_image(self):
        s = self.image
        d0 = ord(s[0])
        d1 = ord(s[1])
        assert len(s)==d0*d1+2,(len(s),d0,d1)
        return numpy.frombuffer(s[2:],dtype='B').reshape(d0,d1)
    def float_image(self):
        return numpy.array(self.byte_image(),'f')/255.0
    def lineparams(self):
        return numpy.array([float(x) for x in self.rel.split()],'f')
    def rel_lineparams(self):
        return docproc.rel_geo_normalize(self.rel)


def chardb(con,rw=1):
    """Open a character database.  This mainly just sets the row factory to CharRow, which
    allows easier access to images."""
    if type(con)==str:
        con = sqlite3.connect(con,timeout=600.0)
    else:
        con = con
    con.row_factory = CharRow
    con.text_factory = str
    return con

def table(con,table,**kw):
    """Ensures that the table exists and that the given columns exist
    in the table; adds columns as necessary.  Columns are specified as
    colname="type" in the argument list."""
    cur = con.cursor()
    cols = list(cur.execute("pragma table_info("+table+")"))
    colnames = [col[1] for col in cols]
    if colnames==[]:
        cmd = "create table "+table+" (id integer primary key"
        if verbose: print "#",cmd
        for k,v in kw.items():
            cmd += ", %s %s"%(k,v)
        cmd += ")"
        if verbose: print "SQL",cmd
        cur.execute(cmd)
    else:
        # table already exists; add any missing columns
        for k,v in kw.items():
            if not k in colnames:
                cmd = "alter table "+table+" add column "+k+" "+v
                if verbose: print "SQL",cmd
                cur.execute(cmd)
    con.commit()
    del cur

def execute(db,query,*params):
    """Execute a query on the connection for side-effects."""
    cur = db.cursor()
    try:
        if verbose: print "SQL",query,params
        cur.execute(query,params)
    finally:
        cur.close()
        del cur

def query(db,query,*params):
    """Perform a simple query on the connection, yielding the rows."""
    cur = db.cursor()
    try:
        if verbose: print "SQL",query,params
        for row in cur.execute(query,params):
            yield row
    finally:
        cur.close()
        del cur

def row_query(db,q,*params):
    """Perform a query on the result yielding a single row as a result (or an error)."""
    result = list(query(db,q,*params))
    assert len(result)==1
    return result[0]

def col_query(db,query,*params):
    """Perform a simple query on the connection; the query should give only
    a single column as a result.  Returns a list of results."""
    cur = db.cursor()
    try:
        if verbose: print "SQL",query,params
        for row in cur.execute(query,params):
            assert len(row)==1
            yield row[0]
    finally:
        cur.close()
        del cur

def tuple_query(db,query,*params):
    """Perform a simple query on the connection and return the result as
    an interator of tuples."""
    cur = db.cursor()
    try:
        if verbose: print "SQL",query,params
        for row in cur.execute(query,params):
            yield tuple(row)
    finally:
        cur.close()
        del cur

def value_query(db,q,*params):
    """Perform a query on the result yielding a single value as a result (or an error)."""
    result = tquery(db,q,*params)
    assert len(result==1),"query yielded more than one result: %s"%(result[:10],)
    result = result[0]
    assert len(result==1),"result row contains more than one element: %s"%(result[:10],)
    return result[0]

def update(db,table,where,*params,**assignments):
    values = []
    setters = ""
    for k,v in assignments.items():
        if setters!="": setters += " , "
        setters += "%s = ?"%k
        if type(v)==numpy.ndarray:
            v = smallimage.pickle(v)
        values.append(v)
    cmd = "update "+table+" set "+setters
    if where!="":
        cmd += " where " + where
    else:
        assert len(params)==0,"extra where parameters given but no 'where' clause: %s"%(params,)
    cur = db.cursor()
    params = list(values)+list(params)
    if verbose: print "SQL",cmd,params
    cur.execute(cmd,params)
    cur.close()
    del cur

def insert(db,table,**assignments):
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
    if verbose: print "SQL",cmd,params
    cur = db.cursor()
    cur.execute(cmd,params)
    cur.close()
    del cur

def ids(db,table):
    return list(col_query(db,"select id from "+table))

def get(db,table,id):
    return row_query(db,"select * from "+table+" where id=?",id)

def put(db,table,id,**kw):
    update(db,table,"id=?",id,**kw)
    
