.SUFFIXES: .c
CFLAGS = -O2

all: $S
.c:
	cc $(CFLAGS) -o $@ $<

S = convert csv validate report encode decode split session offset stride flip center rates pairs replay pack state dp dm tm tp onestep events hawkes decompose compose refill

onestep: onestep.c
	cc $(CFLAGS) -o $@ $< -lm

hawkes: hawkes.c
	cc $(CFLAGS) -o $@ $< -lm

all: $S
clean:
	rm -f $S
