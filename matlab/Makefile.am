if WITH_MATLAB
  CXX=mex
  
  bin_PROGRAMS = dwt_haar
  
  AM_CPPFLAGS = -I$(srcdir)/../include
  LDADD = ../src/libdwt.a
  
  dwt_haar_SOURCES = dwt_haar.cpp
endif
