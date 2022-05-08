CXX := g++ -c
LD := g++
INCLUDES += -Isrc
LIBS +=
CXXFLAGS := $(INCLUDES)
LDFLAGS :=

SRCDIR := src
IMEDIR := $(SRCDIR)/ime
TESTDIR := $(SRCDIR)/test
SRCS := $(wildcard $(IMEDIR)/*.cc)
TEST_SRCS := $(wildcard $(TESTDIR)/*.cc)
TEST_OBJS := $(TEST_SRCS:%.cc=%.o)
OBJS := $(SRCS:%.cc=%.o)
DEPS := $(SRCS:%.cc=%.d) $(TEST_SRCS:%.cc=%.d) $(SRCDIR)/train.d $(SRCDIR)/test.d

.PHONY: all clean debug release

all: release

debug: CXXFLAGS += -g -O0
debug: LDFALGS +=
debug: train test test_dict

release: CXXFLAGS += -O3 -DNDEBUG=1 -fopenmp
release: LDFLAGS += -fopenmp
release: train test test_dict

prof: CXXFLAGS += -O3 -DNDEBUG=1 -pg
prof: LDFLAGS += -pg
prof: train test test_dict

clean:
	rm -rf train test $(OBJS) $(TEST_OBJS) $(SRCDIR)/train.o $(SRCDIR)/test.o $(DEPS)

%.o : %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

%.d : %.cc
	$(CXX) -M $(CXXFLAGS) -o $@ $<

train: $(SRCDIR)/train.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

test: $(SRCDIR)/test.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

test_dict: $(TESTDIR)/test_dict.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

include $(DEPS)