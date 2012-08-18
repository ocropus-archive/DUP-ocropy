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

