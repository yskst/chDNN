#!/usr/bin/python

# Training RBM.

import sys, fileinput
from docopt import docopt

import util,fload

"""Training RBM.
Usage: 
  trainrbm.py [options] <visnum> <hidnum> <file>

options:
   -h, --help   Show this help.
   --of=<file>  output file name.(npz file)
   --df=<str>   sample format flags.The flag's detail is follow.
   --mb=<num>   mini-batch size.
   -e <num> --epoch=<num>   the number of ephoch.
   --lr <val>   learning rate [default: 0] 
   --mm <val>   momentum [default: 0] 
   --re <val>   regulalizer. [default: 0]
   --rt <bb|rb> bb=bernoulli-bernoulli, gb=gaussian-bernoulli.
   --seed <NUM> The seed of random value.[default=1234]
""")

if __name__=='__main__':


