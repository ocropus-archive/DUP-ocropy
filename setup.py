#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys,time,urllib,traceback,glob,os,os.path

assert sys.version_info[0]==2 and sys.version_info[1]>=7,\
    "you must install and use OCRopus with Python version 2.7 or later, but not Python 3.x"

from distutils.core import setup #, Extension, Command
#from distutils.command.install_data import install_data

if not os.path.exists("models/en-default.pyrnn.gz"):
    print()
    print("You must download the default model 'en-default.pyrnn.gz'")
    print("and put it into ./models.")
    print()
    print("Check https://github.com/tmbdev/ocropy for the location")
    print("of model files.")
    print()
    sys.exit(1)

scripts = [c for c in glob.glob("ocropus-*") if "." not in c and "~" not in c]

setup(
    name = 'ocropy',
    version = 'v0.2',
    author = "Thomas Breuel",
    description = "The OCRopy RNN-based Text Line Recognizer",
    packages = ["ocrolib"],
    data_files= [('share/ocropus', ["models/en-default.pyrnn.gz"])],
    scripts = scripts,
    )
