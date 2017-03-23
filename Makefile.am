SUBDIRS = src lib mpi_tests workit check .

test:
	sh check/testc.sh
	sh check/test.sh

install:
	make
	mkdir -p ${prefix}
	echo "Adding mpi.so to ${prefix}"
	cp src/mpi.so ${prefix}
	mkdir -p ${prefix}/mpi_examples
	echo "Installing mpi examples"
	cp -r mpi_tests/* ${prefix}/mpi_examples/
