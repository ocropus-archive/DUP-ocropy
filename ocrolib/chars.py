# -*- encoding: utf-8 -*-
from __future__ import unicode_literals

import re

# common character sets

digits = "0123456789"
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
symbols = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
ascii = digits+letters+symbols  # pylint: disable=redefined-builtin

xsymbols = """€¢£»«›‹÷©®†‡°∙•◦‣¶§÷¡¿▪▫"""
german = "ÄäÖöÜüß"
french = "ÀàÂâÆæÇçÉéÈèÊêËëÎîÏïÔôŒœÙùÛûÜüŸÿ"
turkish = "ĞğŞşıſ"
greek = "ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
portuguese = "ÁÃÌÍÒÓÕÚáãìíòóõú"
telugu = " ఁంఃఅఆఇఈఉఊఋఌఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహఽాిీుూృౄెేైొోౌ్ౘౙౠౡౢౣ౦౧౨౩౪౫౬౭౮౯"

default = ascii+xsymbols+german+french+portuguese

european = default+turkish+greek

# List of regular expressions for normalizing Unicode text.
# Cleans up common homographs. This is mostly used for
# training text.

# Note that the replacement of pretty much all quotes with
# ASCII straight quotes and commas requires some
# postprocessing to figure out which of those symbols
# represent typographic quotes. See `requote`

# TODO: We may want to try to preserve more shape; unfortunately,
# there are lots of inconsistencies between fonts. Generally,
# there seems to be left vs right leaning, and top-heavy vs bottom-heavy

replacements = [
    ('[_~#]', "~"),  # OCR control characters
    ('"', "''"),  # typewriter double quote
    ("`", "'"),  # grave accent
    ('[“”]', "''"),  # fancy quotes
    ("´", "'"),  # acute accent
    ("[‘’]", "'"),  # left single quotation mark
    ("[“”]", "''"),  # right double quotation mark
    ("“", "''"),  # German quotes
    ("„", ",,"),  # German quotes
    ("…", "..."),  # ellipsis
    ("′", "'"),  # prime
    ("″", "''"),  # double prime
    ("‴", "'''"),  # triple prime
    ("〃", "''"),  # ditto mark
    ("µ", "μ"),  # replace micro unit with greek character
    ("[–—]", "-"),  # variant length hyphens
    ("ﬂ", "fl"),  # expand Unicode ligatures
    ("ﬁ", "fi"),
    ("ﬀ", "ff"),
    ("ﬃ", "ffi"),
    ("ﬄ", "ffl"),
]


def requote(s):
    s = re.sub("''", '"', s)
    return s


def requote_fancy(s, germanic=0):
    if germanic:
        # germanic quoting style reverses the shapes
        # straight double quotes
        s = re.sub(r"\s+''", "”", s)
        s = re.sub(r"''\s+", "“", s)
        s = re.sub(r"\s+,,", "„", s)
        # straight single quotes
        s = re.sub(r"\s+'", "’", s)
        s = re.sub(r"'\s+", "‘", s)
        s = re.sub(r"\s+,", "‚", s)
    else:
        # straight double quotes
        s = re.sub(r"\s+''", "“", s)
        s = re.sub(r"''\s+", "”", s)
        s = re.sub(r"\s+,,", "„", s)
        # straight single quotes
        s = re.sub(r"\s+'", "‘", s)
        s = re.sub(r"'\s+", "’", s)
        s = re.sub(r"\s+,", "‚", s)
    return s
