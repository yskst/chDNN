#!/usr/bin/env python

import sys, re, fileinput

def stdout(s):
    sys.stdout.write(s)
def stderr(s):
    sys.stderr.write(s)

def print_help(prog=sys.argv[0]):
    stdout("Splice vector.\n")
    stdout("Usage: %s [options] [files]...\n")
    stdout("Options:\n")
    stdout("  -l, --llen:  The number of left side length.\n")
    stdout("  -r, --rlen:  The number of right side length.\n")
    stdout("  -h, --help: Show this help.\n")

def popn(lst, start, end):
    length = end - start
    for i in range(length):
        lst.pop(start)
    return lst

def print_v(v):
    for e in v:
        stdout("% .9e " % (e))
    stdout('\n')


def parse(s):
    s = re.sub('#.*', '', s) # remove comment
    return map(float, s.split())

def read_and_parse(f, nline=1):
    s = ""
    for i in range(nline):
        s += f.readline()
    return parse(s)

if __name__=='__main__':
    llen = None
    rlen = None
    dim = None
    buf = None
    
    count = 1
    for i,s in enumerate(sys.argv):
        if s == '-l' or s == '--llen':
            llen = int(sys.argv[i+1])
            count += 2
        elif s == '-r' or s == '--rlen':
            rlen = int(sys.argv[i+1])
            count += 2
        elif s == '-h' or s == '--help':
            print_help()
            sys.exit(0)
        
    if (llen is None) or (rlen is None):
        stderr("The right or left side width is not specified.(see --help)\n")
        sys.exit(1)

    f = fileinput.input(sys.argv[count:])
    
    while dim is None:
        v = read_and_parse(f)
        if len(v) == 0: continue  # empty line
        dim = len(v)
    buf = [0.0 for i in range(dim*llen)]
    buf.extend(v)
    buf.extend(read_and_parse(f, rlen))
    print_v(buf)

    for l in f:
        v = parse(l)
        if len(v) == 0: continue
        popn(buf, 0, dim)
        buf.extend(v)
        print_v(buf)

    v = [0.0 for i in range(dim)]
    for i in range(rlen):
        popn(buf,0,dim)
        buf.extend(v)
        print_v(buf)

