.SUFFIXES: .c

.c:
	cc -O2 -Wall -o $@ $<

all: convert validate report encode decode split session offset stride flip center overlap
clean:
	rm -f convert validate report encode decode split session offset stride flip center overlap
