.SUFFIXES: .c

.c:
	cc -O2 -Wall -o $@ $<

all: convert csv validate report encode decode split session offset stride flip center overlap
clean:
	rm -f convert csv validate report encode decode split session offset stride flip center overlap
