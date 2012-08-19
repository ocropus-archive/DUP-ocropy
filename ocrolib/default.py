# the defaults used by the recognizer

model = "en-uw3-linerel.cmodel"
ngraphs = "en-mixed-4.ngraphs"
space = "en-space.model"
lineest = "en-mixed.lineest"

# install the default models

installable = [model,ngraphs,space,lineest]

# an isolated character recognition model

installable += ["en-uw3unlv-perchar.cmodel"]

# gradient based line estimator (script independent)

installable += ["gradient.lineest"]

# another line estimator

installable += ["en-mixed-round1.lineest"]

# these are system paths

modeldir = "/usr/local/share/ocropus/"

import os
def getlocal():
    """Get the path to the local directory where OCRopus data is
    installed. Checks OCROPUS_DATA in the environment first,
    otherwise defaults to /usr/local/share/ocropus."""
    local = os.getenv("OCROPUS_DATA") or modeldir
    return local
