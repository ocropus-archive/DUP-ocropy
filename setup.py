#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

import glob,os
from distutils.core import setup, Extension
from distutils.command.install_data import install_data

class smart_install_data(install_data):
    # not currently used
    def run(self):
        os.system("cd data; for i in *.gz; do gunzip < $i > $(basename $i .gz); done")
        return install_data.run(self)


setup (name = 'ocropy',
       version = '0.1',
       author      = "Thomas Breuel",
       description = """Python bindings for OCRopus""",
       packages = ["ocrolib"],
       data_files=[('share/ocropus', glob.glob("*.glade")),
                   ('share/ocropus', glob.glob("data/*model")),
                   ('share/ocropus', glob.glob("data/*.fst")),
                   ],
       scripts = [i for i in glob.glob("ocropus-*[a-z]") if not i.endswith('.glade')] +
                 glob.glob("ocroex-*[a-z]") +
                 glob.glob("ocrotest-*[a-z]"),
       )
