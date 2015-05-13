from numpy import dtype
import tables
import re

def table_copy(source,dest,names=None,omit=[],verbose=1):
    if names is None:
        names = [name for name in dir(source.root) if re.match(r'^[a-zA-Z][a-zA-Z0-9_][a-zA-Z0-9]*$',name)]
        names = [name for name in names if name not in omit]
    elif isinstance(names, str):
        names = names.split()
    for name in names:
        a = source.getNode("/"+name)
        if verbose: print "[copying",name,a.shape,a.atom,"]"
        if "VLArray" in str(a):
            b = dest.createVLArray(dest.root,name,a.atom,filters=tables.Filters(9))
        else:
            b = dest.createEArray(dest.root,name,a.atom,shape=[0]+list(a.shape[1:]),filters=tables.Filters(9))
        for i in range(len(a)):
            b.append([a[i]])
        dest.flush()
        
def assign_array(db,name,a,verbose=1):
    if a.dtype==dtype('int32'):
        atom = tables.Int32Atom()
    elif a.dtype==dtype('int64'):
        atom = tables.Int64Atom()
    elif a.dtype==dtype('f') or a.dtype==dtype('d'):
        atom = tables.Float32Atom()
    else:
        raise Exception('unknown array type: %s'%a.dtype)
    if verbose: print "[writing",name,a.shape,atom,"]"
    node = db.createEArray(db.root,name,atom,shape=[0]+list(a.shape[1:]),filters=tables.Filters(9))
    node.append(a)

def log_copy(db,odb):
    for name in dir(db.root._v_attrs):
        if name[:4]!="LOG_": continue
        value = db.getNodeAttr("/",name)
        odb.setNodeAttr("/",name,value)

def log(db,*args):
    import time
    db.setNodeAttr("/","LOG_%d"%int(time.time())," ".join(args))

def create_earray(db,name,element_shape,type='f'):
    if type=='f' or type=='float32': atom = tables.Float32Atom()
    elif type=='i' or type=='float64': atom = tables.Int64Atom()
    else: raise Exception("unknown array type; choose one of: 'i', 'f'")
    return db.createEArray(db.root,name,atom,shape=(0,)+tuple(element_shape),filters=tables.Filters(9))
