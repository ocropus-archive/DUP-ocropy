ocropy
======

Python-based OCR package using recurrent neural networks.

To install, use:

    $ sudo apt-get install $(cat PACKAGES)
    $ wget -nd http://www.tmbdev.org/en-default.pyrnn.gz
    $ mv en-default.pyrnn.gz models/
    $ sudo python setup.py install

To test the recognizer, run:

    $ ./run-test

OCRopus is really a collection of document analysis programs, not a turn-key OCR system.

In addition to the recognition scripts themselves, there are a number of scripts for
ground truth editing and correction, measuring error rates, determining confusion matrices, etc.
OCRopus commands will generally print a stack trace along with an error message;
this is not generally indicative of a problem (in a future release, we'll suppress the stack
trace by default since it seems to confuse too many users).

To recognize pages of text, you need to run separate commands: binarization, page layout
analysis, and text line recognition. 

    # perform binarization
    ./ocoropus-nlbin tests/ersch.png -o book

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
