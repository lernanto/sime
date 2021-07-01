CC := g++ -c
LD := g++
INCLUDES += -I.
LIBS +=
CFLAGS := $(INCLUDES)
LDFLAGS :=

SRCDIR := src
IMEDIR := $(SRCDIR)/ime
SRCS := $(wildcard $(IMEDIR)/*.cc)
OBJS := $(SRCS:%.cc=%.o)
DEPS := $(SRCS:%.cc=%.d) $(SRCDIR)/train.d $(SRCDIR)/test.d

.PHONY: all clean debug release

all: release

debug: CFLAGS += -g -O0
debug: LDFALGS +=
debug: train test

release: CFLAGS += -O3 -DNDEBUG=1 -fopenmp
release: LDFLAGS += -fopenmp
release: train test

prof: CFLAGS += -O3 -DNDEBUG=1 -pg
prof: LDFLAGS += -pg
prof: train test

clean:
	rm -rf train test $(OBJS) $(SRCDIR)/train.o $(SRCDIR)/test.o $(DEPS)

%.o : %.cc
	$(CC) $(CFLAGS) -o $@ $<

%.d : %.cc
	$(CC) -M $(CFLAGS) -o $@ $<

train: $(SRCDIR)/train.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

test: $(SRCDIR)/test.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

include $(DEPS)