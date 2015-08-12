#!/usr/bin python

import sys

def stdout(s):
    sys.stdout.write(s)

def stderr(s):
    sys.stderr.write(s)

def panic(s):
    sys.stderr.write(s)
    sys.exit(1)

def linear(x):
    """ The dummy function """
    return x
