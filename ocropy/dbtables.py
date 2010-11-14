import sqlite3
import numpy

class Row:
    pass

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
    def __init__(self,con,factory=Row):
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
    def __init__(self,con,tname,factory=Row,read_only=0):
        self.read_only = read_only
        if type(con)==str:
            self.con = sqlite3.connect(con,timeout=600.0)
        else:
            self.con = con
        self.con.row_factory = sqlite3.Row
        self.con.text_factory = str
        self.tname = tname
        self.converters = {}
        self.columns = None
        self.factory = factory
        self.verbose = 0
        cur = self.con.cursor()
        cur.execute("pragma synchronous=off")
        cur.close(); del cur
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
        if cols==[]:
            assert not self.read_only,"attempting to access table '%s' which doesn't exist"%self.tname
            # table doesn't exist, so create it
            cmd = "create table "+self.tname+" (id integer primary key"
            if self.verbose: print "#",cmd
            for k,v in kw.items():
                cmd += ", %s %s"%(k,v)
            cmd += ")"
            cur.execute(cmd)
        else:
            # table already exists; add any missing columns
            for k,v in kw.items():
                if not k in colnames:
                    assert not self.read_only,"attempting to access column '%s.%s' which doesn't exist"%(self.tname,k)
                    cmd = "alter table "+self.tname+" add column ("+k+" "+v+")"
                    print "###",cmd
                    cur.execute(cmd)
        self.con.commit()
    def del_hash(self,kw,commit=1):
        cur = self.con.cursor()
        cmd = "delete from "+self.tname+" where id>=0"
        values = []
        for k,v in kw.items():
            cmd += " and %s=?"%k
            conv = self.converters.get(k,None)
            if conv is not None: v = conv.pickle(v)
            values += [v]
        cur.execute(cmd,values)
        if commit: self.con.commit()
        cur.close(); del cur
    def commit(self):
        self.con.commit()
    def close(self):
        self.con.close(); del self.con
        self.con = None
    def get_keys(self,which):
        cur = self.con.cursor()
        cmd = "select distinct(%s) from "%which+self.tname
        for row in cur.execute(cmd):
            yield row[0]
        cur.close(); del cur
    def query(self,cmd,values=[],commit=0):
        cur = self.con.cursor()
        for row in cur.execute(cmd,values):
            yield row
        self.con.commit()
        cur.close(); del cur
    def get_hash(self,kw,random_=0,limit_=None):
        cur = self.con.cursor()
        cmd = "select * from "+self.tname+" where id>=0"
        values = []
        for k,v in kw.items():
            cmd += " and %s=?"%k
            conv = self.converters.get(k,None)
            if conv is not None: v = conv.pickle(v)
            values += [v]
        if random_:
            cmd += " order by random()"
        if limit_ is not None:
            cmd += " limit %d"%limit_
        if self.verbose: print "#",cmd,values
        for row in cur.execute(cmd,values):
            result = self.factory()
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
        cur = self.con.cursor()
        if self.columns is None:
            cur.execute("PRAGMA table_info("+self.tname+")")
            self.columns = [col[1] for col in cur]
        if item.has_key("id"):
            cmd = "insert or replace into "+self.tname
        else:
            cmd = "insert into "+self.tname
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
        cur.execute(cmd,values)
        self.con.commit()
        result = cur.lastrowid
        cur.close(); del cur
        return result
    def delete(self,**kw):
        self.del_hash(kw)
    def get(self,**kw):
        random_ = 0
        if "random_" in kw:
            random_ = int(kw["random_"])
            del kw["random_"]
        limit_ = 100000000
        if "limit_" in kw:
            limit_ = int(kw["limit_"])
            del kw["limit_"]
        return self.get_hash(kw,random_=random_,limit_=limit_)
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

class ClusterTable(Table):
    def __init__(self,con,factory=Row,name="clusters"):
        Table.__init__(self,con,name,factory=factory)
        self.converter("image",SmallImage())
class CharTable(Table):
    def __init__(self,con,factory=Row,name="chars"):
        Table.__init__(self,con,name,factory=factory)
        self.converter("image",RLEImage())
    
