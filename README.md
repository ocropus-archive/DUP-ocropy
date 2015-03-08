NB: The `clstm` subproject is now in its own repository at

https://github.com/tmbdev/clstm

ocropy
======

Python-based OCR package using recurrent neural networks.

To install OCRopus dependencies system-wide:
    $ sudo apt-get install $(cat PACKAGES)
    $ wget -nd http://www.tmbdev.net/en-default.pyrnn.gz
    $ mv en-default.pyrnn.gz models/
    $ sudo python setup.py install

Alternatively, dependenices can be installed into a [Python Virtual Environment]
(http://docs.python-guide.org/en/latest/dev/virtualenvs/):
    $ virtualenv ocropus_venv/
    $ source ocropus_venv/bin/source
    $ pip install -r requirements_1.txt
    # tables has some dependencies which must be installed first:
    $ pip install -r requirements_2.txt
    $ wget -nd http://www.tmbdev.net/en-default.pyrnn.gz
    $ mv en-default.pyrnn.gz models/

To test the recognizer, run:

    $ ./run-test

OCRopus is really a collection of document analysis programs, not a turn-key OCR system.

In addition to the recognition scripts themselves, there are a number of scripts for
ground truth editing and correction, measuring error rates, determining confusion matrices, etc.
OCRopus commands will generally print a stack trace along with an error message;
this is not generally indicative of a problem (in a future release, we'll suppress the stack
trace by default since it seems to confuse too many users).

To recognize pages of text, you need to run separate commands: binarization, page layout
analysis, and text line recognition. Here is an example for a page of Fraktur text (German);
you need to download the Fraktur model from tmbdev.net/ocropy/fraktur.pyrnn.gz to run this
example:

    # perform binarization
    ./ocropus-nlbin tests/ersch.png -o book

    # perform page layout analysis
    ./ocropus-gpageseg 'book/????.bin.png'

    # perform text line recognition (on four cores, with a fraktur model)
    ./ocropus-rpred -Q 4 -m models/fraktur.pyrnn.gz 'book/????/??????.bin.png'

    # generate HTML output
    ./ocropus-hocr 'book/????.bin.png' -o ersch.html

    # display the output
    firefox ersch.html

There are some things the currently trained models for ocropus-rpred
will not handle well, largely because they are nearly absent in the
current training data. That includes all-caps text, some special symbols
(including "?"), typewriter fonts, and subscripts/superscripts. This will
be addressed in a future release, and, of course, you are welcome to contribute
new, trained models.

You can also generate training data using ocropus-linegen:

    ocropus-linegen -t tests/tomsawyer.txt -f tests/DejaVuSans.ttf

This will create a directory "linegen/..." containing training data
suitable for training OCRopus with synthetic data.
