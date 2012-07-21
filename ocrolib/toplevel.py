import sys,traceback

def obsolete(f):
    def g(*args,**kw):
        traceback.print_stack()
        sys.stderr.write("\n%s: is obsolete\n\n"%f)
        sys.exit(1)
    return g

