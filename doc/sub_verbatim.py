#! /usr/bin/env python

import sys

doc_fn = sys.argv[1]
lines = file(doc_fn, "r").readlines()

context = {}

idx = 0

outf = sys.stdout

verbmode = False
while idx < len(lines):
    line = lines[idx][:-1]
    if line.strip() == "\\begin{verbatim}":
        verbmode = True
        hadcode = False
    if line.strip() == "\\end{verbatim}":
        verbmode = False
    if verbmode and line.lstrip().startswith(">>>"):
        print>>outf, line
        statement = line
        pre_strip_spaces = strip_spaces = len(statement) - len(statement.lstrip())
        statement = statement[strip_spaces+3:]
        strip_spaces += len(statement) - len(statement.lstrip())
        statement = statement.lstrip()
        while idx+1< len(lines) and lines[idx+1].startswith("..."):
            idx += 1
            line = lines[idx][:-1]
            statement += "\n" + line[3+strip_spaces:]
            print>>outf, line
        place = "%s:%d" % (doc_fn, idx+1)
        try:
            code = compile(statement+"\n", place, "eval")
        except SyntaxError:
            code = compile(statement+"\n", place, "single")

        result = eval(code)

        if result is not None:
            s = pre_strip_spaces*" " +\
                str(result).replace("\n", "%s\n" % (pre_strip_spaces*" "))
            print>>outf, s

        hadcode = True
    elif (not verbmode) or (not hadcode):
        print>>outf, line
    idx += 1

