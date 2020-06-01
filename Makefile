CC := g++ -c
LD := g++
INCLUDES += -I.
LIBS +=
# CFLAGS := $(INCLUDES) -g -O0 -DLOG_LEVEL=LOG_VERBOSE
CFLAGS := $(INCLUDES) -O3 -DNDEBUG=1 -fopenmp
LDFLAGS := -fopenmp

.PHONY: all clean

all: train test

clean:
	rm -rf train test ime.o train.o test.o

train: train.o ime.o
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

test: test.o ime.o
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

ime.o: ime.cc
	$(CC) $(CFLAGS) -o $@ $^

train.o: train.cc
	$(CC) $(CFLAGS) -o $@ $^

test.o: test.cc
	$(CC) $(CFLAGS) -o $@ $^