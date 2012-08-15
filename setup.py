#!/usr/bin/env python

import sys,time,urllib,traceback,glob,os,os.path
from distutils.core import setup, Extension, Command
from distutils.command.install_data import install_data

from ocrolib import default
modeldir = "models/"
modelfiles = [default.model,default.space,default.ngraphs,default.lineest]
modelprefix = "http://iupr1.cs.uni-kl.de/~tmb/ocropus-models/"

class InstallPackagesCommand(Command):
    description = "Install Ubuntu packages necessary for OCRopus."
    user_options = []
    def initialize_options(self): pass
    def finalize_options(self): pass
    def run(self):
        status = os.system("apt-get install python-scipy python-matplotlib python-tables python-sklearn")
        if status!=0: raise Exception("package install failed")

class DownloadCommand(Command):
    description = "Download OCRopus datafiles. (This needs to happen prior to installation.)"
    user_options = []
    def initialize_options(self): pass
    def finalize_options(self): pass
    def run(self):
        print "Starting download of about 500Mbytes of model data."
        time.sleep(3) # give them time to quit
        for m in modelfiles:
            dest = modeldir+m
            if os.path.exists(dest):
                print m,": already downloaded"
                continue
            url = modelprefix+m+".bz2"
            cmd = "curl '%s' | bunzip2 > '%s.temp' && mv '%s.temp' '%s'"%(url,dest,dest,dest)
            print "\n#",cmd,"\n"
            if os.system(cmd)!=0:
                print "download failed"
                sys.exit(1)

setup(
        name = 'ocropy',
        version = '0.6',
        author = "Thomas Breuel",
        description = "The core of the OCRopus OCR system.",
        packages = ["ocrolib"],
        data_files=
            [('share/ocropus', glob.glob("*.glade")),
             ('share/ocropus', [modeldir+m for m in modelfiles])],
        scripts = 
            [i for i in glob.glob("ocropus-*[a-z5]") if not i.endswith('.glade')] +
            glob.glob("ocroex-*[a-z]") +
            glob.glob("ocrotest-*[a-z]") +
            ["ocropus"],
        cmdclass = {
            "download" : DownloadCommand,
            "install_ubuntu_packages" : InstallPackagesCommand,
            }
     )
