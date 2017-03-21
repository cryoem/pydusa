#!/bin/sh

startdir=/Users/hvoicu/ClionProjects/pydusa07/pydusa-1.15-sparx-8/
/Users/hvoicu/SPARX3/installation/bin/mpirun -np 2 /Users/hvoicu/ClionProjects/pydusa07/pydusa-1.15-sparx-8/check/testc >& $startdir/check/myoutputc

dashes="-----------------------------------------------------------------------------"
firstline=`head -n 1 $startdir/check/myoutputc | cat`
testcout=`cat $startdir/check/myoutputc`

if test "$testcout" = "01" || test "$testcout" = "10"; then
  echo Basic MPI C test PASSED
else
  echo Basic MPI C test FAILED

  if test "$firstline" = "$dashes"; then
    cat $startdir/check/myoutputc
  fi
fi

rm -f $startdir/check/myoutputc
