CXX := g++ -c
LD := g++
INCLUDES += -Isrc
LIBS +=
CXXFLAGS := $(INCLUDES)
LDFLAGS :=

SRCDIR := src
IMEDIR := $(SRCDIR)/ime
SRCS := $(wildcard $(IMEDIR)/*.cc)
OBJS := $(SRCS:%.cc=%.o)
DEPS := $(SRCS:%.cc=%.d) $(SRCDIR)/train.d $(SRCDIR)/test.d

.PHONY: all clean debug release

all: release

debug: CXXFLAGS += -g -O0
debug: LDFALGS +=
debug: train test

release: CXXFLAGS += -O3 -DNDEBUG=1 -fopenmp
release: LDFLAGS += -fopenmp
release: train test

prof: CXXFLAGS += -O3 -DNDEBUG=1 -pg
prof: LDFLAGS += -pg
prof: train test

clean:
	rm -rf train test $(OBJS) $(SRCDIR)/train.o $(SRCDIR)/test.o $(DEPS)

%.o : %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

%.d : %.cc
	$(CXX) -M $(CXXFLAGS) -o $@ $<

train: $(SRCDIR)/train.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

test: $(SRCDIR)/test.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

include $(DEPS)