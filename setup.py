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
       scripts = [i for i in glob.glob("ocropus-*") if not i.endswith('.glade')],
       data_files=[('share/ocropus/models',
                    ["2m2-reject.cmodel","multi3.cmodel"]),
                   ('share/ocropus/gui',
                    glob.glob("*.glade"))],
       )
