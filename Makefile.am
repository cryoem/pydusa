SUBDIRS = src lib mpi_tests workit check .

test:
	sh check/testc.sh
	sh check/test.sh

install:
	make
	mkdir -p ${prefix}/lib
	echo "Adding mpi.so to ${prefix}/lib"
	cp src/mpi.so ${prefix}/lib
	mkdir -p ${prefix}/mpi_examples
	echo "Installing mpi examples"
	cp -r mpi_tests/* ${prefix}/mpi_examples/
