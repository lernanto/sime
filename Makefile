CC := g++ -c
LD := g++
INCLUDES += -I.
LIBS +=
CFLAGS := $(INCLUDES) -g -O0
LDFLAGS :=

.PHONY: all clean

all: test

clean:
	rm -rf test test.o ime.o

test: test.o ime.o
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

test.o: test.cc
	$(CC) $(CFLAGS) -o $@ $^

ime.o: ime.cc
	$(CC) $(CFLAGS) -o $@ $^