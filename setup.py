#!/usr/bin/env python

from __future__ import print_function

import sys
import glob
import os.path
from distutils.core import setup

assert sys.version_info[0]==2 and sys.version_info[1]>=7,\
    "you must install and use OCRopus with Python version 2.7 or later, but not Python 3.x"

if not os.path.exists("models/en-default.pyrnn.gz"):
    print()
    print("You should download the default model 'en-default.pyrnn.gz'")
    print("and put it into ./models.")
    print()
    print("Check https://github.com/tmbdev/ocropy for the location")
    print("of model files.")
    print()

models = [c for c in glob.glob("models/*pyrnn.gz")]
scripts = [c for c in glob.glob("ocropus-*") if "." not in c and "~" not in c]

setup(
    name = 'ocropy',
    version = 'v1.0',
    author = "Thomas Breuel",
    description = "The OCRopy RNN-based Text Line Recognizer",
    packages = ["ocrolib"],
    data_files= [('share/ocropus', models)],
    scripts = scripts,
    )
