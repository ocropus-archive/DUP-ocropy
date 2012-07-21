import sys,traceback

def obsolete(f):
    def g(*args,**kw):
        traceback.print_tb()
        sys.stderr("\n%s: is obsolete\n\n"%f)
        sys.exit(1)
