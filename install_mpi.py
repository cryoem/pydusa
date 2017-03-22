#!/usr/bin/env python

def myexec(cmd):
	import os
	import sys
	print  "	 ", cmd
	r = os.system(cmd)
	if r != 0:
		print "Command execution failed!"
		print "If it failed at wget or curl due to no Internet connection, try download the file from other machine,",
		print "copy it to the current directory and restart install_mpi.py.",
		print "Otherwise, check the log file and try to resolve it."
		sys.exit(-1)


def chdir(dir):
	import os
	print  "	  cd ", dir
	os.chdir(dir)


def macos():
	import commands
	r = commands.getoutput("uname")
	return r=="Darwin"


def geturl(url, file):
	import os, sys
	import commands

	if macos():
		myexec("curl " + url + " -o " + file)
	else:
		myexec("wget " + url)
	# sys.exit(0)

def get_mpiroot(options):
	import os
	import commands
	if options.force:
		return False
	print "Checking mpicc"
		
	r = os.system("mpicc --version")
	if r != 0:
		print "Cannot find mpicc"
		return False
		
	return True
	

def get_numpyroot(pythonroot):
	try:
		import numpy
		return pythonroot 

	except Exception, inst:
		print "problem import numpy:", inst
		return None


def install_fftw3_mpi():
	import os

	pwd = os.getcwd()
	if os.path.exists(pwd + "/fftw_mpi/installation"):
		return
	chdir("fftw_mpi/fftw-3.3.5")
	myexec("mkdir %s"%(pwd + "/fftw_mpi/installation"))
	myexec("./configure --prefix=%s --enable-mpi --enable-shared"%(pwd + "/fftw_mpi/installation"))
	# myexec("./configure --prefix=%s --enable-mpi --enable-shared"%(eman2))
	myexec("make clean")
	myexec("make")
	myexec("make install")

	os.chdir(pwd)
	return

def install_openmpi(version):
	import os
	root = os.environ['EMAN2DIR']

	pwd = os.getcwd()

	file = "openmpi-%s.tar.gz"%version
	major_version = version.split(".")
	major_version =  major_version[0] + "." + major_version[1]
	if macos():
		url = " https://www.open-mpi.org/software/ompi/v%s/downloads/"%major_version  + file
	else:
		url = " --no-check-certificate  https://www.open-mpi.org/software/ompi/v%s/downloads/"%major_version  + file
	
	print ""
	print "Installing openmpi-%s"%version
	if not(os.path.exists("openmpi-%s"%version)) :
		if not(os.path.exists(file)):
			geturl(url, file)
		myexec("tar -zxf " + file)
		
	chdir("openmpi-%s"%version)
	myexec("./configure --enable-static --prefix=" + root + " --disable-dlopen")
	myexec("make clean")
	myexec("make")
	myexec("make install")
	os.chdir(pwd)
	return root

def update_Makefile_src():
	import os

	pwd = os.getcwd()
	chdir("src")

	library_location = "%s/fftw_mpi/installation/lib"%pwd
	adding_dict = {
		"CFLAGS = " : ' -I%s/fftw_mpi/installation/include -DPYDUSA_VERSION=%s'%(pwd, pwd),
		"LDFLAGS = " : " -L" + library_location + " -lfftw3_mpi -lfftw3 -lm "
	}

	statbuf = os.stat("Makefile")

	move_to_original = True
	with open("Makefile", "r") as fp, open("Makefile___out", "w") as fp_out:
		for line in fp:
			# if "-g -O2" in line:
				# line = line.replace("-g -O2", "-g -O1")
			for key in adding_dict:
				if line[:len(key)] == key:
					if adding_dict[key] not in line:
						fp_out.write(line[:-1])
						fp_out.write(adding_dict[key])
						fp_out.write("\n")
					else:
						fp_out.write(line)
						move_to_original = False
					break
			else:
				fp_out.write(line)
				
	if move_to_original:
		myexec("mv Makefile Makefile___original")
		myexec("mv Makefile___out Makefile")

	os.utime("Makefile",(statbuf.st_atime,statbuf.st_mtime))

	eman2_source_file = ""
	if os.path.exists(os.environ["EMAN2DIR"] + os.sep + "eman2.bashrc"):
		eman2_source_file = os.environ["EMAN2DIR"] + os.sep + "eman2.bashrc"
		my_list = open(eman2_source_file).readlines()
		if my_list.count("export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH\n"%library_location) == 0:
			my_list.append("export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH\n" % library_location)
			open(eman2_source_file, "w").writelines(my_list)
	elif os.path.exists(os.environ["EMAN2DIR"] + os.sep + "eman2.cshrc"):
		eman2_source_file = os.environ["EMAN2DIR"] + os.sep + "eman2.cshrc"
		my_list = open(eman2_source_file).readlines()
		if my_list.count("setenv LD_LIBRARY_PATH %s:${LD_LIBRARY_PATH}\n"%library_location) == 0:
			my_list.append("setenv LD_LIBRARY_PATH %s:${LD_LIBRARY_PATH}\n" % library_location)
			open(eman2_source_file, "w").writelines(my_list)
	elif os.path.exists(os.environ["EMAN2DIR"] + os.sep + "eman2.zshrc"):
		eman2_source_file = os.environ["EMAN2DIR"] + os.sep + "eman2.zshrc"
		my_list = open(eman2_source_file).readlines()
		if my_list.count("export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH\n"%library_location) == 0:
			my_list.append("export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH\n" % library_location)
			open(eman2_source_file, "w").writelines(my_list)

	os.chdir(pwd)
	return eman2_source_file, "export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH\n"%library_location



from optparse import OptionParser
import string

default_version_of_open_mpi_to_istall = "1.10.2"

usage = "install_mpi.py --force --openmpi_ver=%s"%default_version_of_open_mpi_to_istall
parser = OptionParser(usage)
parser.add_option( "--force", action="store_true", default=False, help="forcefully install necessary packages" )
parser.add_option("--openmpi_ver", type="string",  default=default_version_of_open_mpi_to_istall, help="version of openmpi to forcefully install, default = %s"%default_version_of_open_mpi_to_istall)
options,args = parser.parse_args()

import os
from sys import exit
import commands

try:
	eman2 = os.environ['EMAN2DIR']
except:
	print "Error: cannot find EMAN2DIR environment variable, cannot proceed! You have to install EMAN2 first."
	exit(-1)

try:
	# Try to fix problem with dependencies caused by EMAN2 - sometimes it is needed to make configure scripts work
	ld_library_path = os.environ["LD_LIBRARY_PATH"]
	os.environ["LD_LIBRARY_PATH"] = string.replace(ld_library_path, eman2 + "/lib:", "")
except:
	pass

path_to_python = eman2 + "/Python/bin/python"

#pythonroot, python_version = get_pythonroot(options)
if not get_mpiroot(options):
	if options.force:
		print ""
		print "=====> Install OpenMPI version %s"%options.openmpi_ver
		install_openmpi(options.openmpi_ver)
	else:
		print "You need MPI environment (both runtime and developer packages) and gcc compiler to continue. "
		print "If you work on professional HPC cluster, in all likelihood both are already installed. "
		print "In this case read the user guide - you have to probably load appriopriate module by \"module load\" command."
		print "You can also run this script again with the --force option - it will download and install MPI (openmpi-%s) for you."%default_version_of_open_mpi_to_istall
		exit(-1)

## need to install fftw3-mpi
install_fftw3_mpi()

print ""
print "=====> Configuring the mpi python binding"

# if os.path.exists(eman2 + "/Python"):
# 	myexec("./configure --with-python=" + eman2 + "/Python/bin/python --prefix=" + eman2)
# else:
# 	# EMAN2 doesn't have Python - default one will be used
# 	myexec("./configure --prefix=" + eman2)

# if options.ls5:
# 	if os.path.exists(eman2 + "/Python"):
# 		myexec("MPILIB=$TACC_CRAY_MPT_LIB  ./configure --with-mpicc=/opt/apps/intel16/cray_mpich/7.3.0/bin/mpicc --with-python=" + eman2 + "/Python/bin/python --prefix=" + eman2)
# 	else:
# 		#	EMAN2 doesn't have Python - default one will be used
# 		myexec("MPILIB=$TACC_CRAY_MPT_LIB  ./configure --with-mpicc=/opt/apps/intel16/cray_mpich/7.3.0/bin/mpicc --prefix=" + eman2)
# else:
if os.path.exists(eman2 + "/Python"):
	myexec("./configure --with-python=" + eman2 + "/Python/bin/python --prefix=" + eman2)
else:
	# EMAN2 doesn't have Python - default one will be used
	myexec("./configure --prefix=" + eman2)


##  need to update the Makefile in src to include the -I and -L for fftw-mpi compilation
eman2_source_file, bash_command_to_add = update_Makefile_src()
		
print ""
print "=====> Building the mpi python binding"
myexec("make clean >> log.txt")	
myexec("make all >> log.txt")


print ""
print "=====> Install the mpi python binding"
myexec("make install >> log.txt")	


if macos():
	print
	print "=====> Start SPARX or EMAN2 and run import mpi. If it runs without any error message the installation is complete."
	print
else:
	if eman2_source_file != "":
		print "=====> To complete installation you need to run:"
		print "source %s"%eman2_source_file
	else:
		print "Unkown shell. You will need set up your enviroment variables manually:"
		print bash_command_to_add

	print
	print "Then start SPARX or EMAN2 and run 'import mpi'. If it runs without any error message the installation is complete."
	print

