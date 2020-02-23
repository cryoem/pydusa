#! /usr/bin/env /home2/stevel/miniconda3/envs/e3/bin/python
import sys
import Numeric
from Numeric import *
import string
from string import replace
from datetime import datetime
import mpi
from mpi import *
import subProcess


def handle(st,count,command,mystdio,mystderr):
    global f,nstart
    if(count==nstart):
        outname="output"+"%4.4d" % myid
        f=open(outname, 'w')
    f.write(str(st))
    f.write("\n")
    stamp=datetime.today()
    f.write(str(stamp))
    f.write(" COMMAND:\n")
    f.write(command)
    f.flush()
    f.write("\nSTDOUT:\n")
    f.write(mystdio)
    f.write("\nSTDERR:\n")
    f.write(mystderr)
    f.write("\n\n")
    f.flush()

def mypack(x):
    i=len(x)
    back=zeros(i,"i")
    for j in range(0,i):
        back[j]=ord(x[j:j+1])
    return back

def myunpack(x):
    i=len(x)
    back=""
    for j in range(0,i):
        back=back+chr(x[j])
    return back

#start mpi and get the number of processes
sys.argv =  mpi_init(len(sys.argv),sys.argv)
myid=mpi_comm_rank(MPI_COMM_WORLD)
numprocs=mpi_comm_size(MPI_COMM_WORLD)
print "hello form ",myid," of ",numprocs
#define some useful constants
lastproc=numprocs-1
id_p1=myid+1
id_m1=myid-1
my_tag=12345

# process 0 tells the others how many commands to run
if(myid == 0):
    print sys.argv
    nc=len(sys.argv)-1
    nstart=1
    if(sys.argv[1] == "-s"):
        nstart=2
else:
    nc=0
    nstart=0
nc=mpi_bcast(nc,1,MPI_INT,0,MPI_COMM_WORLD)
nstart=mpi_bcast(nstart,1,MPI_INT,0,MPI_COMM_WORLD)

if(nc == 0):
	print "nothing to do"
	mpi_finalize()
	sys.exit(0)

# get the base command for each process
#fname="com"+"%4.4d" % myid
#f12=open(fname, 'r')
#com=f12.readline()
if(myid == 0):
    f12=open("commands",'r')
    com=f12.readline()
    for ic in range(1,numprocs):
        comic=f12.readline()
        if(len(comic) == 0):
            comic="echo i have nothing to do with DUMMY"
        comray=mypack(comic)
        thelen=len(comray)
        mpi_send(comray,thelen, MPI_INT,ic,my_tag,MPI_COMM_WORLD)
else:
        mpi_probe(0,my_tag,MPI_COMM_WORLD)
        icount=mpi_get_count(MPI_INT)
        comray=mpi_recv(icount,MPI_INT,0,my_tag,MPI_COMM_WORLD)
        com=myunpack(comray)

if(myid == 0):
    for ic in range(nstart,nc+1):
        # get real command string
        fname=sys.argv[ic]
        if(nstart == 2):
            p=fname.rfind(".")
            if(p > 0):
                fname=fname[0:p]
        rc=replace(com, "DUMMY", fname)
        # do command
        print "task",(ic-nstart)+1," started at",datetime.today()," on ",fname
        st=datetime.today()
        process=subProcess.subProcess(rc)
        process.read()
        handle(st,ic,rc,process.outdata,process.errdata)
        del(process)
        # wait for ready from process 1
        i=mpi_recv(1,MPI_INT,id_p1,my_tag,MPI_COMM_WORLD)
        # send filename to process 1
        fnameray=mypack(fname)
        thelen=len(fnameray)
        mpi_send(fnameray,thelen, MPI_INT,id_p1,my_tag,MPI_COMM_WORLD)
else:
    for ic in range(nstart,nc+1):
        #send ready to previous process
        i=1
        mpi_send(i,1, MPI_INT,id_m1,my_tag,MPI_COMM_WORLD)
        #get filename from previous process
        mpi_probe(id_m1,my_tag,MPI_COMM_WORLD)
        icount=mpi_get_count(MPI_INT)
        fnameray=mpi_recv(icount,MPI_INT,id_m1,my_tag,MPI_COMM_WORLD)
        fname=myunpack(fnameray)
        rc=replace(com, "DUMMY", fname)
        # do command
        st=datetime.today()
        process=subProcess.subProcess(rc)
        process.read()
        handle(st,ic,rc,process.outdata,process.errdata)
        del(process)
        if(myid != lastproc):
            # wait for ready next process
            i=mpi_recv(1,MPI_INT,id_p1,my_tag,MPI_COMM_WORLD)
            # send filename to next process
            mpi_send(fnameray,icount, MPI_INT,id_p1,my_tag,MPI_COMM_WORLD)
        else:
            print "task",(ic-nstart)+1," completed at",datetime.today()," on ",fname
        
mpi_barrier(MPI_COMM_WORLD)
mpi_finalize()
