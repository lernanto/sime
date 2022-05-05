CXX = cl.exe /c
LINK = link
INCLUDES = /I src
LIBS =
CXXFLAGS = $(INCLUDES) /utf-8 /O2 /DNDEBUG=1 /openmp
LINKFLAGS =

SRCDIR = src
IMEDIR = $(SRCDIR)\ime
OBJS = $(IMEDIR)\dict.obj $(IMEDIR)\model.obj $(IMEDIR)\decoder.obj

all: release

release: train.exe test.exe

clean:
	del train.exe test.exe $(OBJS) $(SRCDIR)\train.obj $(SRCDIR)\test.obj

.cc.obj:
	$(CXX) $(CXXFLAGS) /Fo"$@" $**

train.exe: $(SRCDIR)\train.obj $(OBJS)
	$(LINK) $(LINKFLAGS) /out:train.exe $** $(LIBS)

test.exe: $(SRCDIR)\test.obj $(OBJS)
	$(LINK) $(LINKFLAGS) /out:test.exe $** $(LIBS)