import os

with open("results1.txt", "w") as fp:
    # for num_test in range(60, 150):
    for num_test in range(192,193):
        print 4*num_test+3
        if os.system(" mpirun -np 16 ./a.out %d"%(4*num_test+3)):
        # if os.system(" mpirun -np 14 ./a.out %d"%(4*num_test+3)):
            fp.write("%d\n"%(4*num_test+3))
            fp.flush()
# /Users/hvoicu/ClionProjects/pydusa10/pydusa-1.15-sparx-8/test_fftw3d.py