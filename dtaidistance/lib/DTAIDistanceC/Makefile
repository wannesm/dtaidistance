
CC=gcc
# This works for macos, on linux it can be -lgomp needs to be used
CFLAGS=-Wall -g -Xpreprocessor -fopenmp -lomp
CPPFLAGS=
DEPS = DTAIDistanceC/dd_globals.h
OBJ = DTAIDistanceC/dd_benchmark.o \
	  DTAIDistanceC/dd_dtw.o \
	  DTAIDistanceC/dd_dtw_openmp.o \
	  DTAIDistanceC/dd_ed.o


default: dd_benchmark


DTAIDistanceC/%.o: DTAIDistanceC/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

dd_benchmark: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)



clean:
	rm -f DTAIDistanceC/*.o
	rm -f dd_benchmark






