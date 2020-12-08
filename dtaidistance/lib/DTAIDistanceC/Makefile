
CC=gcc
CFLAGS=-Wall -g
CPPFLAGS=
DEPS = DTAIDistanceC/dd_globals.h
OBJ = DTAIDistanceC/dd_benchmark.o \
	  DTAIDistanceC/dd_dtw.o \
	  DTAIDistanceC/dd_dtw_openmp.o \
	  DTAIDistanceC/dd_ed.o


default: dtaidistancec


DTAIDistanceC/%.o: DTAIDistanceC/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

dd_benchmark: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)



clean:
	rm -f DTAIDistanceC/*.o
	rm -f dd_benchmark






