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
AS_IF([test x$with_matlab != no],
      [MEX_PATH="$with_matlab/bin:$MEX_PATH"]
      [AC_PATH_PROG([MEX], [mex], [no], [$MEX_PATH])]
      [AS_IF([test x$MEX != xno],
             [HAVEMATLAB=yes]
             [AC_SUBST(MEX)]
             [MATLAB_ROOT="$with_matlab"]
             [AC_SUBST(MATLAB_ROOT)],
             [with_matlab=no])
      ],
      [with_matlab=no])
AM_CONDITIONAL([WITH_MATLAB], [test "x$with_matlab" = xyes])
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
echo "        With matlab:     $with_matlab"
if test x$with_matlab == xyes; then
echo "        Matlab path:     $MATLAB_ROOT"
fi
echo ""
echo "        Verbosity level: $messaging"

echo ""
echo ""
