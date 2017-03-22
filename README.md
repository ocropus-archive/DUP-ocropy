ocropy
======

[![Build Status](https://travis-ci.org/tmbdev/ocropy.svg)](https://travis-ci.org/tmbdev/ocropy)
[![license](https://img.shields.io/github/license/tmbdev/ocropy.svg)](https://github.com/tmbdev/ocropy/blob/master/LICENSE)
[![Wiki](https://img.shields.io/badge/wiki-9%20pages-orange.svg)](https://github.com/tmbdev/ocropy/wiki)
[![Join the chat at https://gitter.im/tmbdev/ocropy](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/tmbdev/ocropy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

OCRopus is a collection of document analysis programs, not a turn-key OCR system.
In order to apply it to your documents, you may need to do some image preprocessing,
and possibly also train new models.

In addition to the recognition scripts themselves, there are a number of scripts for
ground truth editing and correction, measuring error rates, determining confusion matrices, etc.
OCRopus commands will generally print a stack trace along with an error message;
this is not generally indicative of a problem (in a future release, we'll suppress the stack
trace by default since it seems to confuse too many users).

Installing
----------

To install OCRopus dependencies system-wide:

    $ sudo apt-get install $(cat PACKAGES)
    $ wget -nd http://www.tmbdev.net/en-default.pyrnn.gz
    $ mv en-default.pyrnn.gz models/
    $ sudo python setup.py install

Alternatively, dependencies can be installed into a
[Python Virtual Environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/):

    $ virtualenv ocropus_venv/
    $ source ocropus_venv/bin/activate
    $ pip install -r requirements.txt
    $ wget -nd http://www.tmbdev.net/en-default.pyrnn.gz
    $ mv en-default.pyrnn.gz models/
    $ python setup.py install

An additional method using [Conda](http://conda.pydata.org/) is also possible:

    $ conda create -n ocropus_env python=2.7
    $ source activate ocropus_env
    $ conda install --file requirements.txt
    $ wget -nd http://www.tmbdev.net/en-default.pyrnn.gz
    $ mv en-default.pyrnn.gz models/
    $ python setup.py install

To test the recognizer, run:

    $ ./run-test

Running
-------

To recognize pages of text, you need to run separate commands: binarization, page layout
analysis, and text line recognition. The default parameters and settings of OCRopus assume
300dpi binary black-on-white images. If your images are scanned at a different resolution, the
simplest thing to do is to downscale/upscale them to 300dpi. The text line recognizer is
fairly robust to different resolutions, but the layout analysis is quite resolution dependent.

Here is an example for a page of Fraktur text (German);
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

## Roadmap

------------------------
| Project Announcements
|:-----------------------
| The text line recognizer has been ported to C++ and is now a separate project, the CLSTM project, available here: https://github.com/tmbdev/clstm
| Please welcome @zuphilip and @kba as additional project maintainers. @tmb is busy developing new DNN models for document analysis (among other things). (10/15/2016)
------------------------

A lot of excellent packages have become available for deep learning, vision, and GPU computing over the last few years.
At the same time, it has become feasible now to address problems like layout analysis and text line following
through attentional and reinforcement learning mechanisms. I (@tmb) am planning on developing new software using these
new tools and techniques for the traditional document analysis tasks. These will become available as separate
projects.

Note that for text line recognition and language modeling, you can also use the CLSTM command line tools. Except for taking different command line options, they are otherwise drop-in replacements for the Python-based text line recognizer.

## Contributing

OCRopy and CLSTM are both command line driven programs. The best way to contribute is to create new command line programs using the same (simple) persistent representations as the rest of OCRopus.

The biggest needs are in the following areas:

 - text/image segmentation
 - text line detection and extraction
 - output generation (hOCR and hOCR-to-* transformations)

## CLSTM vs OCRopy

The CLSTM project (https://github.com/tmbdev/clstm) is a replacement for 
`ocropus-rtrain` and `ocropus-rpred` in C++ (it used to be a subproject of
`ocropy` but has been moved into a separate project now). It is significantly faster than 
the Python versions and has minimal library dependencies, so it is suitable 
for embedding into C++ programs.

Python and C++ models can _not_ be interchanged, both because the save file 
formats are different and because the text line normalization is slightly 
different. Error rates are about the same.

In addition, the C++ command line tool (`clstmctc`) has different command line 
options and currently requires loading training data into HDF5 files, instead
of being trained off a list of image files directly (image file-based training
will be added to `clstmctc` soon).

The CLSTM project also provides LSTM-based language modeling that works very
well with post-processing and correcting OCR output, as well as solving a number
of other OCR-related tasks, such as dehyphenation or changes in orthography
(see our publications). You can train language models using `clstmtext`.

Generally, your best bet for CLSTM and OCRopy is to rely only on the command
line tools; that makes it easy to replace different components. In addition, you
should keep your OCR training data in .png/.gt.txt files so that you can easily 
retrain models as better recognizers become available.

After making CLSTM a full replacement for `ocropus-rtrain`/`ocropus-rpred`, the
next step will be to replace the binarization, text/image segmentation, and layout 
analysis in OCRopus with trainable 2D LSTM models.

## Solution for clang

[Read README_OSX.md](README_OSX.md)
