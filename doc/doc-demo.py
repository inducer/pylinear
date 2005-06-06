import sys
import StringIO
import traceback

lines = file(sys.argv[1], "r").readlines()

i = 0
in_verbatim = False

exec_globals = {}

while i < len(lines):
    lines[i] = lines[i].rstrip()
    if lines[i].startswith("\\begin{verbatim}"):
        in_verbatim = True
    if lines[i].startswith("\\end{verbatim}"):
        in_verbatim = False
    if lines[i].startswith(">>>"):
        command = lines[i][3:].rstrip()
        lendiff = len(command) - len(command.lstrip())
        command = command.lstrip()
        while i+1 < len(lines) and lines[i+1].startswith("..."):
            i += 1
            command += "\n"+lines[i][3+lendiff:].rstrip()
        try:
            result = repr(eval(compile(command, "a", "single")))
            if result is not None:
                lines[i+1:i+1] = result.split("\n")
        except:
            print >> sys.stderr, "*** ERROR"
            outbuf = StringIO.StringIO()
            traceback.print_exc(file = outbuf)
            result = outbuf.getvalue().rstrip()
            lines[i+1:i+1] = result.split("\n")
    i += 1



for l in lines:
    print l
