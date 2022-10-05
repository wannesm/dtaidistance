# Typical paths for OpenMP.
# Include:
# /usr/local/include
# /opt/local/include/libomp
# /opt/homebrew/include
#
# Library:
# /usr/local/lib
# /opt/local/lib/libomp
# /opt/homebrew/lib/

CC=gcc
# CC=/opt/homebrew/opt/llvm/bin/clang
# This works for macos, on linux it can be -lgomp needs to be used
# MacPorts:
# CFLAGS=-Wall -g -Xpreprocessor -fopenmp -lomp -I/opt/local/include/libomp  -L/opt/local/lib/libomp 
# HomeBrew:
# CFLAGS=-Wall -g -Xpreprocessor -fopenmp -lomp -I/opt/homebrew/include -L/opt/homebrew/lib
CFLAGS=-Wall -g -Xpreprocessor -fopenmp -lomp -I/opt/homebrew/include
# CFLAGS="-Wall -g -Xpreprocessor -fopenmp -lomp  -I/opt/homebrew/opt/llvm/include -I/opt/homebrew/include"
# LDFLAGS="-L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/lib"
LDFLAGS=/opt/homebrew/lib/libomp.a -Wall -g 
CPPFLAGS=
DEPS = DTAIDistanceC/dd_globals.h jinja
OBJ = DTAIDistanceC/dd_benchmark.o \
	  DTAIDistanceC/dd_dtw.o \
	  DTAIDistanceC/dd_dtw_openmp.o \
	  DTAIDistanceC/dd_ed.o


default: dd_benchmark


DTAIDistanceC/%.o: DTAIDistanceC/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

dd_benchmark: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)



clean:
	rm -f DTAIDistanceC/*.o
	rm -f dd_benchmark



