# the defaults used by the recognizer

rnnmodel = "en-default.pyrnn.gz"
model = "en-uw3-linerel-2.cmodel.gz"
ngraphs = "en-mixed-3.ngraphs.gz"
space = "en-space.model.gz"
lineest = "en-mixed.lineest.gz"

# install the default models

installable = [rnnmodel,model,ngraphs,space,lineest]
installable += ["uw3unlv.pyrnn.gz"]
installable += ["en-uw3unlv-perchar.cmodel.gz"] # isolated character model
installable += ["gradient.lineest.gz"] # gradient based line model
installable += ["en-mixed-round1.lineest.gz"] # another line estimator
installable += ["frakant.pyrnn.gz"] # Fraktur recognizer
installable += ["fraktur.pyrnn.gz"] # Fraktur recognizer

modeldir = "/usr/local/share/ocropus/"

import os
def getlocal():
    """Get the path to the local directory where OCRopus data is
    installed. Checks OCROPUS_DATA in the environment first,
    otherwise defaults to /usr/local/share/ocropus."""
    local = os.getenv("OCROPUS_DATA") or modeldir
    return local
