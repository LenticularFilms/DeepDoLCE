CC=gcc

CFLAGS=-Wall -g -std=c11 -O3

LDFLAGS=-ltiff -liberty

doLCE: lenticularFilmColorReconstruction.o imageHandling.o imageProcessing.o auxilaries.o
	cc $(CFLAGS) $(LDFLAGS) lenticularFilmColorReconstruction.o imageHandling.o imageProcessing.o auxilaries.o -o doLCE

imageHandling.o: imageHandling.c imageHandling.h
	cc -c imageHandling.c

imageProcessing.o: imageProcessing.c imageProcessing.h
	cc -c imageProcessing.c

auxilaries.o: auxilaries.c auxilaries.h
	cc -c auxilaries.c


clean:
	rm imageHandling.o lenticularFilmColorReconstruction.o imageProcessing.o auxilaries.o doLCE

install:
	cp doLCE /usr/local/bin/

uninstall: 
	rm /usr/local/bin/doLCE

testing:
	cp doLCE ../1_testing-devel-version
