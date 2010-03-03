#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

import glob
from distutils.core import setup, Extension

setup (name = 'ocropy',
       version = '0.1',
       author      = "Thomas Breuel",
       description = """Python bindings for OCRopus""",
       packages = ["ocropy"],
       scripts = glob.glob("ocropus-*"),
       data_files=[('/usr/local/share/ocropus/models',
                    ["2m2-reject.cmodel","multi3.cmodel"]),
                   ('/usr/local/share/ocropus/gui',
                    ["ocropus-cedit.glade"])],
       )
