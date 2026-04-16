.SUFFIXES: .c

.c:
	cc -O2 -Wall -o $@ $<

all: convert encode decode split session
clean:
	rm -f convert encode decode split session
