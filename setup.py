#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

import glob,os,os.path
from distutils.core import setup, Extension
from distutils.command.install_data import install_data

assert os.path.exists("models/space.model"),\
    "you must download models first (cd models; ./ocropus-download)"

setup (name = 'ocropy',
       version = '0.5',
       author      = "Thomas Breuel",
       description = """Python bindings for OCRopus""",
       packages = ["ocrolib"],
       data_files=[('share/ocropus', glob.glob("*.glade")),
                   ('share/ocropus', glob.glob("models/*")),
                   ],
       scripts = [i for i in glob.glob("ocropus-*[a-z5]") if not i.endswith('.glade')] +
                 glob.glob("ocroex-*[a-z]") +
                 glob.glob("ocrotest-*[a-z]") +
                 ["ocropus"],
       )
