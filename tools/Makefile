.SUFFIXES: .c

.c:
	cc -O2 -Wall -o $@ $<

all: convert encode decode sessions
clean:
	rm -f convert encode decode
