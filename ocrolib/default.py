# the defaults used by the recognizer

import os

modeldir = "/usr/local/share/ocropus/"

def getlocal():
    """Get the path to the local directory where OCRopus data is
    installed. Checks OCROPUS_DATA in the environment first,
    otherwise defaults to /usr/local/share/ocropus."""
    local = os.getenv("OCROPUS_DATA") or modeldir
    return local

traceback = int(os.getenv("OCROTRACE") or "0")
