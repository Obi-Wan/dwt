AC_INIT([dwt], [0.1], [ben.vighy@gmail.com])

AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall dist-bzip2])

AC_PROG_CXX
AC_PROG_RANLIB
m4_ifdef([AM_PROG_AR], [AM_PROG_AR])

AC_CONFIG_HEADERS([src/config.h])
AC_CONFIG_FILES([
  Makefile
  include/Makefile
  src/Makefile
  matlab/Makefile
  examples/Makefile
])

CFLAGS="${CFLAGS=}"
CXXFLAGS="${CXXFLAGS=}"
AC_SUBST([CFLAGS],["${CFLAGS} -Wall"])
AC_SUBST([CXXFLAGS],["${CXXFLAGS} -Wall"])

my_save_cflags="$CFLAGS"
CFLAGS="-std=c++11"
AC_MSG_CHECKING([whether CC supports the flag -std=c++11])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([])],
    [AC_MSG_RESULT([yes])]
    [CPP_11_FLAG=-std=c++11],
    [AC_MSG_RESULT([no])]
    [CPP_11_FLAG=-std=c++0x]
)
CFLAGS="$my_save_cflags"
AC_SUBST([CXXFLAGS],["${CXXFLAGS} ${CPP_11_FLAG}"])

###############################################################################
# Definitions

dnl CHECK_MEX_SUFFIX(list-of-suffices, variable-to-set)
AC_DEFUN([CHECK_MEX_SUFFIX],[
cat >conftest.cc <<_ACEOF
extern "C" void mexFunction() {
}
_ACEOF
$CXX -fPIC -c -o conftest.o conftest.cc
$MEX -cxx -output conftest conftest.o
$2=""
for suffix in $1; do
  if test -f "conftest.$suffix"; then
    $2="$suffix"
    rm -f "conftest.$suffix"
  fi
done
rm -f conftest.cc conftest.o
])



###############################################################################
# Debug builds
AC_ARG_WITH(messaging,
  AS_HELP_STRING([--with-messaging=level],
                 [enable various messaging levels, default: 1 (info)
                  Options: 3 (debug), 2 (warnings), 1 (info), 0 (no messages)]),
  [ case "${withval}" in
      3)     messaging=debug ;;
      2)     messaging=warn ;;
      1)     messaging=info ;;
      0)     messaging=nomessage ;;
      false) messaging=nomessage ;;
      no)    messaging=nomessage ;;
      *)   AC_MSG_ERROR([bad value ${withval} for --with-messaging]) ;;
    esac],
  [messaging=info])
AS_IF([test "x$messaging" == xdebug],
  [AC_DEFINE([DEBUG], [1],
        [Define if you want a debug build])
   ])
AS_IF([test "x$messaging" == xwarn],
  [AC_DEFINE([WARNING], [1],
        [Define if you want a build with warning information])
   ])
AS_IF([test "x$messaging" == xinfo],
  [AC_DEFINE([INFO], [1],
        [Define if you want a build with information])
   ])

AS_IF([test "x$CXX" == xclang++],
  [AC_SUBST([CXXFLAGS],["${CXXFLAGS} -stdlib=libc++"])
   ])

AC_ARG_WITH(matlab,
  AS_HELP_STRING([--with-matlab=<path>],
                 [compiles matlab's mex interface]),
  [],
  [with_matlab=no])
MATLAB_PATH=$PATH
HAVE_MATLAB=no
AS_IF([test x$with_matlab != xno],
      [MEX_PATH="$with_matlab/bin:$PATH"]
      [AC_PATH_PROG([MEX], [mex], [no], [$MEX_PATH])]
      [AS_IF([test x$MEX != xno],
             [HAVE_MATLAB=yes]
             [AC_SUBST(MEX)]
             [MATLAB_ROOT="$with_matlab"]
             [AC_SUBST(MATLAB_ROOT)]
             [with_matlab=yes]
             [CHECK_MEX_SUFFIX([mexa64 mexglx mexmaci64 mexmaci],[MEXSUFFIX])]
             [AS_IF([test x$MEXSUFFIX != x],
                    [AC_SUBST(MEXSUFFIX)],
                    [with_matlab=no]
                    )
             ],
             [with_matlab=no]
             )
      ],
      [with_matlab=no]
      )
AC_SUBST(HAVE_MATLAB)

AC_ARG_WITH(octave,
  AS_HELP_STRING([--with-octave=<path>],
                 [compiles octave's oct interface]),
  [],
  [with_octave=no])
OCTAVE_PATH=$PATH
HAVE_OCTAVE=no
AS_IF([test x$with_octave != xno],
      [OCT_PATH="$with_octave/bin:$PATH"]
      [AC_PATH_PROG([MKOCTFILE], [mkoctfile], [no], [$OCT_PATH])]
      [AS_IF([test x$MKOCTFILE != xno],
             [HAVE_OCTAVE=yes]
             [AC_SUBST(MKOCTFILE)]
             [with_octave=yes],
             [with_octave=no]
             )
      ],
      [with_octave=no]
      )
AC_SUBST(HAVE_OCTAVE)
#AM_CONDITIONAL([WITH_OCTAVE], [test "x$with_octave" = xyes])

AC_ARG_ENABLE([vector],
    AS_HELP_STRING([--disable-vector], [Disable vectorization extensions])
)
AS_IF([test "x$enable_vector" != "xno"],
    [AC_DEFINE([USE_VECTORIZATION], [1], [Define if you want vectorization extensions to be used])]
    [enable_vector=yes])

AC_ARG_ENABLE([openmp],
    AS_HELP_STRING([--disable-openmp], [Disable OpenMP parallelization support])
)
AS_IF([test "x$enable_openmp" != "xno"],
    [AC_DEFINE([HAVE_OMP], [1], [Define if you want OpenMP parallelization to be used])]
    [enable_openmp=yes]
    [AC_SUBST([CXXFLAGS],["${CXXFLAGS} -fopenmp"])]
    [AC_SUBST([LDFLAGS],["${LDFLAGS} -fopenmp"])] )
AC_SUBST([enable_openmp])

#AS_IF([test "x$messaging" != xdebug],
#      [ AC_SUBST([CFLAGS],["${CFLAGS} -fomit-frame-pointer -fforce-addr"])
#        AC_SUBST([CXXFLAGS],["${CXXFLAGS} -fomit-frame-pointer -fforce-addr"])
#       ])
###############################################################################

AC_OUTPUT

dnl
dnl Output some configuration info for the user
dnl
echo ""
echo "  Configuration summary:"
echo ""
echo "        prefix:          $prefix"
echo "        exec_prefix:     $exec_prefix"
echo "        libdir:          $libdir"
echo "        includedir:      $includedir"

dnl
dnl Output the features included
dnl
echo ""
echo "        CFLAGS:          $CFLAGS"
echo "        CXXFLAGS:        $CXXFLAGS"
echo ""
echo "        Vectorization:   $enable_vector"
echo ""
echo "        OpenMP:          $enable_openmp"
echo ""
echo "        With octave:     $with_octave"
if test x$with_octave == xyes; then
echo "        Mkoctfile path:  $MKOCTFILE"
fi
echo ""
echo "        With matlab:     $with_matlab"
if test x$with_matlab == xyes; then
echo "        Matlab path:     $MATLAB_ROOT"
echo "        Mex suffix:      $MEXSUFFIX"
fi
echo ""
echo "        Verbosity level: $messaging"

echo ""
echo ""
