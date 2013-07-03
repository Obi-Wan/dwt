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
])

CFLAGS="${CFLAGS=}"
CXXFLAGS="${CXXFLAGS=}"
AC_SUBST([CFLAGS],["${CFLAGS} -Wall"])
AC_SUBST([CXXFLAGS],["${CXXFLAGS} -Wall -std=c++0x"])

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
echo "        Verbosity level: $messaging"

echo ""
echo ""
