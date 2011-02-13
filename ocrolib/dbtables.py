import os
import sqlite3
import numpy
import docproc
import utils

debug = os.getenv("dbtables_debug")
if debug!=None: debug = int(debug)

class SmallImage:
    def pickle(self,image):
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
    def unpickle(self,s):
        d0 = ord(s[0])
        d1 = ord(s[1])
        assert len(s)==d0*d1+2,(len(s),d0,d1)
        return numpy.frombuffer(s[2:],dtype='B').reshape(d0,d1)

class RLEImage:
    def pickle(self,image):
        w,h = image.shape
        assert w<=255
        assert h<=255
        a = image.ravel()
        i = 0
        n = len(a)
        result = [w,h]
        threshold = (amax(a)+amin(a))/2
        while i<n:
            start = i
            while i<n and i-start<250 and a[i]<threshold: i+=1
            if i<n: result.append(i-start)
            start = i
            while i<n and i-start<250 and a[i]>=threshold: i+=1
            if i<n: result.append(i-start)
        return array(result,'B')
    def unpickle(self,s):
        w = ord(s[0])
        h = ord(s[1])
        a = zeros(w*h,'B')
        i = 0
        val = 0
        for c in s[2:]:
            for j in range(ord(c)):
                a[i] = val
                i += 1
            val = 255-val
        a.shape = (w,h)
        return a
    
class Database:
    def __init__(self,con,factory=sqlite3.Row):
        if type(con)==str:
            self.con = sqlite3.connect(con,timeout=600.0)
        else:
            self.con = con
        self.con.row_factory = sqlite3.Row
        self.con.text_factory = str
        self.factory = factory
        cur = self.con.cursor()
        cur.execute("pragma synchronous=off")
        cur.close(); del cur
    def query(self,cmd,values=[],commit=0):
        cur = self.con.cursor()
        for row in cur.execute(cmd,values):
            yield row
        self.con.commit()
        cur.close(); del cur

class Table:
    def __init__(self,con,tname,factory=sqlite3.Row,read_only=0):
        self.ignore_extra_keys = 0
        self.read_only = read_only
        if type(con)==str:
            self.con = sqlite3.connect(con,timeout=600.0)
        else:
            self.con = con
        print self.con
        self.con.row_factory = factory
        self.con.text_factory = str
        self.tname = tname
        self.converters = {}
        self.columns = None
        self.factory = factory
        self.verbose = 0
        cur = self.con.cursor()
        cur.execute("pragma synchronous=off")
        cur.close(); del cur
    def __enter__(self,):
        return self
    def __exit__(self,type,value,traceback):
        self.close()
        return None
    def converter(self,cname,conv):
        self.converters[cname] = conv
    def read(self,ignore=1,**kw):
        cur = self.con.cursor()
        cols = list(cur.execute("pragma table_info("+self.tname+")"))
        colnames = [col[1] for col in cols]
        for k,v in kw.items():
            assert k in colnames,"expected column %s missing from database"%k
        cur.close()
    def create(self,ignore=1,**kw):
        return self.open(ignore=ignore,**kw)
    def open(self,ignore=1,**kw):
        cur = self.con.cursor()
        cols = list(cur.execute("pragma table_info("+self.tname+")"))
        colnames = [col[1] for col in cols]
        self.columns = colnames
        if cols==[]:
            assert not self.read_only,"attempting to access table '%s' which doesn't exist"%self.tname
            # table doesn't exist, so create it
            cmd = "create table "+self.tname+" (id integer primary key"
            if self.verbose: print "#",cmd
            for k,v in kw.items():
                cmd += ", %s %s"%(k,v)
                self.columns.append(k)
            cmd += ")"
            if debug: print cmd
            cur.execute(cmd)
        else:
            # table already exists; add any missing columns
            for k,v in kw.items():
                if not k in colnames:
                    assert not self.read_only,"attempting to access column '%s.%s' which doesn't exist"%(self.tname,k)
                    cmd = "alter table "+self.tname+" add column "+k+" "+v
                    print "###",cmd
                    if debug: print cmd
                    cur.execute(cmd)
                    self.columns.append(k)
        self.con.commit()
        del cur
    def del_hash(self,kw,commit=1):
        cur = self.con.cursor()
        cmd = "delete from "+self.tname+" where id>=0"
        values = []
        for k,v in kw.items():
            cmd += " and %s=?"%k
            conv = self.converters.get(k,None)
            if conv is not None: v = conv.pickle(v)
            values += [v]
        if debug: print cmd
        cur.execute(cmd,values)
        if commit: self.con.commit()
        cur.close(); del cur
    def commit(self):
        self.con.commit()
    def close(self):
        self.con.close()
        del self.con
        self.con = None
    def get_keys(self,which):
        cur = self.con.cursor()
        cmd = "select distinct(%s) from "%which+self.tname
        if debug: print cmd
        for row in cur.execute(cmd):
            yield row[0]
        cur.close(); del cur
    def count(self):
        cmd = "select count(*) from "+self.tname
        cur = self.con.cursor()
        for row in cur.execute(cmd):
            result = int(row[0])
            break
        cur.close(); del cur
        return result
    def query(self,cmd,values=[],commit=0):
        cur = self.con.cursor()
        if debug: print cmd
        for row in cur.execute(cmd,values):
            yield row
        self.con.commit()
        cur.close(); del cur
    def get_hash(self,kw,random_=0,limit_=None,sample_=None):
        cur = self.con.cursor()
        cmd = "select * from "+self.tname+" where id>=0"
        values = []
        for k,v in kw.items():
            cmd += " and %s=?"%k
            conv = self.converters.get(k,None)
            if conv is not None: v = conv.pickle(v)
            values += [v]
        if sample_:
            cmd += " and id%%%d=abs(random())%%%d"%(sample_,sample_)
        if random_:
            cmd += " order by random()"
        if limit_ is not None:
            cmd += " limit %d"%limit_
        if self.verbose: print "#",cmd,values
        if debug: print cmd
        for row in cur.execute(cmd,values):
            result = utils.Record()
            for k in row.keys():
                conv = self.converters.get(k,None)
                v = row[k]
                if conv is not None:
                    v = conv.unpickle(v)
                setattr(result,k,v)
            yield result
        cur.close(); del cur
    def put_hash(self,item,commit=1):
        assert len(item.keys())>0
        assert self.columns is not None,"call open first"
        cur = self.con.cursor()
        if item.has_key("id"):
            cmd = "insert or replace into "+self.tname
        else:
            cmd = "insert into "+self.tname
        if not self.ignore_extra_keys:
            for key in item.keys():
                assert key in self.columns,"key %s missing from columns %s"%(key,self.columns)
        names = [x for x in item.keys() if x in self.columns]
        cmd += " ("+",".join(names)+")"
        cmd += " values ("+",".join(["?"]*len(names))+")"
        values = []
        for k in names:
            v = item[k]
            conv = self.converters.get(k,None)
            if conv is not None: v = conv.pickle(v)
            values += [v]
        if self.verbose: print "#",cmd,values
        if debug: print cmd,values[:1]+values[2:]
        cur.execute(cmd,values)
        self.con.commit()
        result = cur.lastrowid
        cur.close(); del cur
        return result
    def delete(self,**kw):
        self.del_hash(kw)
    def get(self,**kw):
        random_ = 0
        sample_ = None
        limit_ = 100000000
        if "sample_" in kw:
            sample_ = int(kw["sample_"])
            del kw["sample_"]
        if "random_" in kw:
            random_ = int(kw["random_"])
            del kw["random_"]
        if "limit_" in kw:
            limit_ = int(kw["limit_"])
            del kw["limit_"]
        return self.get_hash(kw,random_=random_,limit_=limit_,sample_=sample_)
    def set(self,**kw):
        return self.put_hash(kw)
    def put(self,x,commit=1):
        if type(x)==dict:
            return self.put_hash(x,commit=commit)
        elif type(x)==list:
            result = [self.put(object,commit=0) for object in x]
            if commit: self.con.commit()
            return result
        else:
            return self.put_hash(x.__dict__,commit=commit)
    def execute(self,cmd,values=[],commit=0):
        cur = self.con.cursor()
        if debug: print cmd
        for row in cur.execute(cmd,values):
            yield row
        self.con.commit()
        cur.close(); del cur
    def keys(self):
        return self.find("id>=0")
    def items(self):
        for row in self.execute("select * from '%s'"%self.tname):
            yield row["id"],row
    def find(self,expr,values=[]):
        query = "select id from '%s' where "%self.tname+expr
        ids = [row[0] for row in self.execute(query,values)]
        return ids
    def __getitem__(self,index):
        rows = self.execute("select * from '%s' where id=?"%self.tname,[index])
        result = list(rows)
        assert len(result)==1
        return result[0]

class ClusterTable(Table):
    def __init__(self,con,factory=sqlite3.Row,name="clusters"):
        Table.__init__(self,con,name,factory=factory)
        self.converter("image",SmallImage())

class CharTable(Table):
    def __init__(self,con,factory=sqlite3.Row,name="chars"):
        Table.__init__(self,con,name,factory=factory)
        self.converter("image",RLEImage())

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

class OcroTable(Table):
    def __init__(self,con,name="chars"):
        Table.__init__(self,con,name,factory=CharRow)
