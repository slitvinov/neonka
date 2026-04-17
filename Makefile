.SUFFIXES: .c
CFLAGS = -O2

.c:
	cc $(CFLAGS) -o $@ $<

S = convert csv validate report encode decode split session offset stride flip center rates pairs replay pack

all: $S
clean:
	rm -f $S
