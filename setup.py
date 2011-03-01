#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

import glob,os
from distutils.core import setup, Extension
from distutils.command.install_data import install_data

class smart_install_data(install_data):
    def run(self):
        os.system("cd data; for i in *.gz; do gunzip < $i > $(basename $i .gz); done")
        return install_data.run(self)


setup (name = 'ocropy',
       version = '0.1',
       author      = "Thomas Breuel",
       description = """Python bindings for OCRopus""",
       packages = ["ocrolib"],
       scripts = [i for i in glob.glob("ocropus-*[a-z]") if not i.endswith('.glade')] +
                 glob.glob("ocroex-*[a-z]") +
                 glob.glob("ocrotest-*[a-z]"),
       data_files=[('share/ocropus/gui', glob.glob("*.glade")),
                   ('share/ocropus/models', glob.glob("data/*model")),
                   ('share/ocropus/models', glob.glob("data/*.fst")),
                   ],
       cmdclass = {'install_data': smart_install_data},
       )
