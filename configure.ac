AC_INIT(src/mympimodule.c)
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE(MYMPI, 1.1.5)
AC_PROG_INSTALL

AC_PREFIX_DEFAULT(${HOME}/MYMPI-1.1.5)


AC_PROG_CC
AC_PROG_F77
AC_LANG(C)

AC_CHECK_LIB([m], [main])
AC_PROG_RANLIB

AC_CHECK_SIZEOF([long int])

if test "$ac_cv_sizeof_long_int" = "8"; then
  CFLAGS="$CFLAGS -DLONG64=long"
fi


dashes="--------------------------------------------------------------------------"
currdir=`pwd`

AC_ARG_WITH(python,
  [  --with-python=PYTHONPATH           Specify Python location.],
  [PYTHONPATH=$withval],
  [PYTHONPATH=""])

if test "$PYTHONPATH" = "" ; then
  AC_PATH_PROG(py_path, python)
  if test -z "${py_path}"; then
     AC_MSG_ERROR([Unable to find python!  Python location can be specified\
 with --with-python])
  fi
else
  py_path=$PYTHONPATH
fi

AC_MSG_CHECKING([for python include directory])
changequote(<<, >>)dnl
PY_VERSION=`$py_path -c 'import sys; print sys.version[:3]'`
changequote([, ])dnl
PY_PREFIX=`$py_path -c 'import sys; print sys.prefix'`
if test -f $PY_PREFIX/include/python$PY_VERSION/Python.h; then
   PY_HEADER="-I$PY_PREFIX/include/python$PY_VERSION"
   AC_MSG_RESULT([$PY_HEADER])
else
   AC_MSG_ERROR([Unable to find Python.h!])
fi


py_com=`${py_path} -c "import Numeric"`
if test "$?" != "0"; then
        AC_MSG_ERROR([python module Numeric is not installed])
fi

numpy_found="yes"

py_com=`${py_path} -c "import numpy"`
if test "$?" != "0"; then
        echo "(python module numpy is not installed or cannot be imported)"
	numpy_found="no"
fi

AC_ARG_WITH(mpicc, [AC_HELP_STRING([--with-mpicc],[MPI C compiler path])],
                  with_mpicc=$withval, with_mpicc=no)

if test "$with_mpicc" = "no"; then
  AC_PATH_PROG(MPICC, mpicc) 
  if test -z "${MPICC}"; then
    AC_MSG_ERROR([Unable to find mpicc! mpicc location can be specified with --with-mpicc])
  fi
else
  MPICC=$with_mpicc
fi

MPIBIN=`dirname $MPICC`
MPIRUN=${MPIBIN}/mpirun
MPIROOT=`dirname $MPIBIN`

if test -z ${MPIINC}; then
  MPIINC=${MPIROOT}/include
fi

if test -e ${MPIINC}/mpi.h; then
  echo "mpi.h is found"
else
  AC_MSG_ERROR(["mpi.h is not found in the guessed MPI include directory ${MPIINC}.  Please specify the include directory using MPIINC or try another --with-mpicc"])
fi

if test -z ${MPILIB}; then
  find_mpilib=`find ${MPIROOT}/lib* -name libmpi* | head -n 1`
  if test -n ${find_mpilib}; then
    MPILIB=`dirname ${find_mpilib}`
  else
    AC_MSG_ERROR([Unable to find mpi libraries in MPI lib directory.  Please specify the library directory using MPILIB])
  fi
fi

if test -e ${MPILIB}/libmpi.a || test -e ${MPILIB}/libmpi.so || test -e ${MPILIB}/libmpi.dylib; then
  MPILINK="-L${MPILIB} -lmpi"
  LAM=true
elif test -e ${MPILIB}/libmpich.a || test -e ${MPILIB}/libmpich.so || test -e ${MPILIB}/libmpich.dylib; then
  MPILINK="-L${MPILIB} -lmpich"
  MPICH=true
else
  AC_MSG_ERROR([Unable to find libmpi.a, libmpi.so, libmpich.a, or libmpich.so in MPI lib directory])
fi

CC=${MPICC}
CFLAGS="${CFLAGS} -fPIC -I${MPIINC} ${PY_HEADER}"
LDFLAGS="-shared ${MPILINK}"

if test `uname -s` = "Darwin"; then
  LDFLAGS="${LDFLAGS} -undefined dynamic_lookup"
fi

#mpilink1=`dirname $MPICC`
#mpilink=`dirname $mpilink1`/lib

#$MPICC --showme:compile >& tempmpicompile
#$MPICC --showme:link >& tempmpilink
#MPICOMPILE=`cat tempmpicompile` 
#MPILINK=`cat tempmpilink`

#rm -f tempmpicompile tempmpilink


#if test `uname` = "Darwin"; then
#  CFLAGS="$CFLAGS -fno-strict-aliasing -Wno-long-double -no-cpp-precomp -mno-fused-madd -fno-common -dynamic \
#-DNDEBUG -DNULL_INIT -g -Os -Wall -Wstrict-prototypes -DMACOSX -DENABLE_DTRACE -arch `uname -m`"
#  LDFLAGS="$LDFLAGS -Wl,-F. -bundle -undefined dynamic_lookup -arch `uname -m`"
#elif test "$PROCTYPE" = "i686" ; then
#else
#  PROCTYPE=`uname -p`
#  HWPLATFORM=`uname -i`
#
#  if test "$PROCTYPE" = "i686" && test "$HWPLATFORM" = "i386"; then
#    CFLAGS="$CFLAGS -fPIC -march=i386 -mcpu=i686 -D_GNU_SOURCE "
#  fi
#fi


#CFLAGS="$CFLAGS $MPICOMPILE $PY_HEADER"
#LDFLAGS="-shared $LDFLAGS $MPILINK"
#LDFLAGS="-shared -L$mpilink -lmpich "

sed -e 's%@mpirun@%'${MPIRUN}'%' check/test.sh.in > check/test.sh.in.temp
sed -e 's%@startdir@%'${currdir}'%' check/test.sh.in.temp > check/test.sh
sed -e 's%@mpirun@%'${MPIRUN}'%' check/testc.sh.in > check/testc.sh.in.temp
sed -e 's%@startdir@%'${currdir}'%' check/testc.sh.in.temp > check/testc.sh
rm -f check/test.sh.in.temp

# MPI FORTRAn

AC_MSG_CHECKING([for --enable-fortran])

AC_ARG_ENABLE(fortran,
     [AC_HELP_STRING([--enable-fortran], [Enable MPI Fortran compilation])],
     [ with_fortran="enable" ],
     [ with_fortran="disable" ]
)

if test "$with_fortran" = "enable"; then
  AC_MSG_RESULT([yes])
  MPIFC=${MPIBIN}/mpif90

  if test -e ${MPIFC}; then
    echo "mpif90 is found"
  else
    AC_MSG_ERROR([Unable to locate mpif90 in ${MPIBIN} directory])
  fi
  
  $MPIFC mpi_tests/f_ex00.f >& tempmpicompile
  firstline=`head -n 1 tempmpicompile | cat`
 
  if test "$firstline" = "$dashes"; then
    cat tempmpicompile 
    rm -f tempmpicompile
    AC_MSG_ERROR([mpif90 failed to compile a test MPI Fortran program])
  else
    FFLAGS="${FFLAGS} -I${MPIINC}"
    FLDFLAGS="${LDFLAGS} ${MPILINK}"
    rm -f tempmpicompile f_ex00.o a.out fmpi.mod
  fi
else
  AC_MSG_RESULT([no])
fi

#if test "$mpif90_found" = "yes" && test "$mpif90_bad" = "no"; then
#  FFLAGS="$FFLAGS $MPICOMPILE $PY_HEADER"
#  FLDFLAGS="$FLDFLAGS $MPILINK"
#else
#  FFLAGS="$FFLAGS $PY_HEADER"
#fi

AM_CONDITIONAL(COMPILE_FORTRAN, test "$with_fortran" = "enable")

sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex00.py.in > mpi_tests/p_ex00.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex01.py.in > mpi_tests/p_ex01.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex02.py.in > mpi_tests/p_ex02.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex04.py.in > mpi_tests/p_ex04.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex05.py.in > mpi_tests/p_ex05.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex05d.py.in > mpi_tests/p_ex05d.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex06.py.in > mpi_tests/p_ex06.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex07.py.in > mpi_tests/p_ex07.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex08.py.in > mpi_tests/p_ex08.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex09.py.in > mpi_tests/p_ex09.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex10.py.in > mpi_tests/p_ex10.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex12.py.in > mpi_tests/p_ex12.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/p_ex13.py.in > mpi_tests/p_ex13.py

sed -e 's%@python@%'${py_path}'%' mpi_tests/mpi2Examples/main0.py.in > mpi_tests/mpi2Examples/main0.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/mpi2Examples/main1.py.in > mpi_tests/mpi2Examples/main1.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/mpi2Examples/tspawn.py.in > mpi_tests/mpi2Examples/tspawn.py
sed -e 's%@python@%'${py_path}'%' mpi_tests/mpi2Examples/worker.py.in > mpi_tests/mpi2Examples/worker.py

sed -e 's%@python@%'${py_path}'%' workit/workit.py.in > workit/workit.py

chmod +x mpi_tests/*.py
chmod +x mpi_tests/mpi2Examples/*.py
chmod +x workit/*.py

AC_SUBST(COMPILE_FORTRAN)
AC_SUBST(MPIROOT)
AC_SUBST(MPIBIN)
AC_SUBST(MPIINC)
AC_SUBST(MPILIB)
AC_SUBST(MPICC)
AC_SUBST(MPIFC)
AC_SUBST(CFLAGS)
AC_SUBST(FFLAGS)
AC_SUBST(LDFLAGS)
AC_SUBST(FLDFLAGS)

AC_OUTPUT([Makefile
           src/Makefile
	   lib/Makefile
	   mpi_tests/Makefile
	   workit/Makefile
	   check/Makefile])

echo ""
echo "     ========================"
echo "     Configuration Parameters"
echo "     ========================"
echo ""
echo "Python                   ${py_path}"
echo "MPI C Compiler:          ${MPICC}"
echo "MPIRUN:                  ${MPIRUN}"
echo "CFLAGS:                  ${CFLAGS}"
echo "LDFLAGS:                 ${LDFLAGS}"

if test "$with_fortran" = "enable"; then
  echo "MPI FORTRAN 90 Compiler: ${MPIFC}"
  echo "FFLAGS:                  ${FFLAGS}"
  echo "FLDFLAGS:                ${FLDFLAGS}"
fi
  
if test "$numpy_found" = "no"; then
  echo ""
  echo "Warning: numpy is not installed in ${py_path} or cannot be imported."
  echo "You cannot run the examples in mpi_tests/mpi2Examples until you have installed numpy correctly."
fi
