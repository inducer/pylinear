#! /bin/sh

set -e

F77=g77
SOURCE_FILES="daux.f ddaskr.f dlinpk.f"
LIBRARY=libdaskr.a

rm -f $LIBRARY

for i in $SOURCE_FILES; do
  DESTNAME=${i%.f}.o
  $F77 -c -o $DESTNAME $i
  ar -rc $LIBRARY $DESTNAME
done

ranlib $LIBRARY
