#!/usr/bin/env python

import sys

from ocrolib import edist, utils

# Test the levenshtein function and returns 0 if the computed value
# equals the one it should be, otherwise returns 1 for failed tests.
def testLevenshtein(a, b, should):
    if edist.levenshtein(a, b) == should:
        print 'ok - levenshtein(%s, %s) == %s' % (a,b,should)
        return 0
    else:
        print 'not ok - levenshtein(%s, %s) == %s' % (a,b,should)
        return 1


def testXLevenshtein(a, b, context, should):
    #print(edist.xlevenshtein(a, b, context))
    if edist.xlevenshtein(a, b, context) == should:
        print 'ok - xlevenshtein(%s, %s, %s) == %s' % (a,b,context,should)
        return 0
    else:
        print 'not ok - xlevenshtein(%s, %s, %s) == %s' % (a,b,context,should)
        return 1


failed_tests = 0

print('# 1 Test function "levenshtein" in edist.py')
failed_tests += testLevenshtein('a', 'a', 0)
failed_tests += testLevenshtein('', '', 0)
failed_tests += testLevenshtein('a', '', 1)
failed_tests += testLevenshtein('', 'a', 1)
failed_tests += testLevenshtein('aa', 'aaaaaa', 4)
failed_tests += testLevenshtein('aba', 'bab', 2)

print('\n# 2 Test function "xlevenshtein" in edist.py')
failed_tests += testXLevenshtein('exccpt', 'except', 1, should=(1.0, [('ccp', 'cep')]))
failed_tests += testXLevenshtein('exccpt', 'except', 2, should=(1.0, [('xccpt', 'xcept')]))
failed_tests += testXLevenshtein('exccpt', 'except', 3, should=(1.0, [('exccpt ', 'except ')]))
failed_tests += testXLevenshtein('exccpt', 'except', 4, should=(1.0, [(' exccpt  ', ' except  ')]))
failed_tests += testXLevenshtein('', 'test', 1, should=(4.0, []))
failed_tests += testXLevenshtein('aaaaaaaaaaa', 'a', 1, should=(10.0, [('aaaaaaaaaaa ', 'a__________ ')]))
failed_tests += testXLevenshtein('123 111 456', '132 111 444', 1, should=(4.0, [('123_ ', '1_32 '), ('456 ', '444 ')]))

print('\n# 3 utils.sumouter / utils.sumprod')
from pylab import randn
utils.sumouter(randn(10,3),randn(10,4),out=randn(3,4))
print('ok - dimensions of sumouter')
utils.sumprod(randn(11,7),randn(11,7),out=randn(7))
print('ok - dimensions of sumprod')

sys.exit(failed_tests)
