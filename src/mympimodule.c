/*
Copyright (c) 2005 The Regents of the University of California
All Rights Reserved
Permission to use, copy, modify and distribute any part of this
	Continuity 6 for educational, research and non-profit purposes,
	without fee, and without a written agreement is hereby granted,
	provided that the above copyright notice, this paragraph and the
	following three paragraphs appear in all copies.
Those desiring to incorporate this software into commercial products or
	use for commercial purposes should contact the Technology
	Transfer & Intellectual Property Services, University of
	California, San Diego, 9500 Gilman Drive, Mail Code 0910, La
	Jolla, CA 92093-0910, Ph: (858) 534-5815, FAX: (858) 534-7345,
	E-MAIL:invent@ucsd.edu.
IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY
	PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
	CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
	THE USE OF THIS SOFTWARE  EVEN IF THE UNIVERSITY OF
	CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
THE SOFTWARE PROVIDED HEREIN IS ON AN AS IS BASIS, AND THE
	UNIVERSITY OF CALIFORNIA HAS NO OBLIGATION TO PROVIDE
	MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
	THE UNIVERSITY OF CALIFORNIA MAKES NO REPRESENTATIONS AND
	EXTENDS NO WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESS,
	INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
	MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE
	USE OF THE Continuity WILL NOT INFRINGE ANY PATENT, TRADEMARK OR
	OTHER RIGHTS.";
*/

/************** numpy or Numeric? **************/

/* if NUMPY is defined then we look for arrayobject.h
   in <arrayobject.h>.  If not then in <Numeric/arrayobject.h>

   The second case would be the norm if you are linking against
   Numeric instead of numpy.

   We default to Numeric but this might change in the future.

   You can see the which was used at run time by printing
   mpi.ARRAY_LIBRARY.

*/
/* !!! CHANGED - SET NUMPY !!! */
#define NUMPY

/************** numpy or Numeric **************/
char DATE_SRC[]="$Date$";
char URL_SRC[]="$HeadURL$";
char REV_SRC[]="$Revision$";
#define VERSION "1.15.0"
#define COPYWRITE "Copyright (c) 2005 The Regents of the University of California All Rights Reserved.  print mpi.copywrite() for details."
#undef DO_UNSIGED
#define DATA_TYPE long
#define COM_TYPE long
#define ARG_ARRAY
/* #define ARG_STR */
/* #define SIZE_RANK */

#include <Python.h>
#include "documentation.h"

#ifdef DOSLU
#include "solvers.h"
#endif

#ifdef NUMPY
#include <numpy/arrayobject.h>
#define LIBRARY "NUMPY"
#else
#include <Numeric/arrayobject.h>
#define LIBRARY "Numeric"
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Toshio Moriya 2018/04/25
// Now let's use C way of error handling! (Toshio 2018/04/25)
#include <errno.h>  // errno; 
// Toshio Moriya 2018/04/25
// We must programatically liking the fftw3 related libraries 
// to solve the DLL hell of fftw functions between fftw3 and NumPy libraries...
#include <dlfcn.h>  // Dl_info, dladdr, dlerror, dlopen, dlsym

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

#ifdef LAM
#define NULL_INIT
#endif

// NOTE: Toshio Moriya 2018/05/08
// Trying to avoid the library conflicts of the same name functions (i.e. fftw_execute(), fftw_destroy_plan()),
// including fftw3-mpi header before NumPy headers did not work.  
// In stead, changing the order causes undefined errors of some symbols (e.g. ‘Dl_info’ undeclared ).
// #include <mpi.h>  # This is included in "fftw3-mpi.h"
#include "fftw3-mpi.h"

/*
static int calls = 0;
static int errs = 0;
static MPI_Comm mycomm;
*/
MPI_Errhandler newerr;

// // Toshio Moriya 2018/04/25
// // Create exception class for not implemented functions
// // #include <new>          // std::bad_alloc
// // #include <exception>    // std::logic_error
// class MRKNotImplemented : public std::logic_error
// {
// 	public:
// 		MRKNotImplemented() : std::logic_error("MRK_DEBUG: Function not yet implemented") { };
// };





#ifdef DEBUG
static FILE *debug;
char fname[50],pname[MPI_MAX_PROCESSOR_NAME];
void writeit(PyArrayObject *result,int count, MPI_Datatype datatype,char *routine) {
	int mycount,i;
	float *df;
	double *dd;
	int *di;
	if(result->nd > 0) {
		mycount=result->dimensions[0];
	}
	else {
		mycount=0;
	}
	fprintf(debug,"from %s %d %d\n",routine,count,mycount);
	if(mycount > 0){
		if(datatype == (MPI_Datatype)MPI_INT) {
			fprintf(debug,"MPI_INT\n");
			di=(int*)result->data;
			for(i=0;i<mycount;i++) {
				fprintf(debug,"%d\n",*di);
				di++;
			}
		}
		if(datatype == (MPI_Datatype)MPI_FLOAT) {
			fprintf(debug,"MPI_FLOAT\n");
			df=(float*)result->data;
			for(i=0;i<mycount;i++) {
				fprintf(debug,"%g\n",*df);
				df++;
			}
		}
		if(datatype == (MPI_Datatype)MPI_DOUBLE) {
			fprintf(debug,"MPI_DOUBLE\n");
			dd=(double*)result->data;
			for(i=0;i<mycount;i++) {
				fprintf(debug,"%g\n",*dd);
				dd++;
			}
		}
	}
	fflush(debug);
}
void dummy(char *routine) {
	fprintf(debug,"from %s empty\n",routine);
	fflush(debug);
}

#endif

#ifdef MPI_VERSION
#if (MPI_VERSION >= 2)
#define MPI2
#endif
#else
#define MPI_VERSION 1
#endif

#ifndef MPI_SUBVERSION
#define MPI_SUBVERSION 0
#endif

char cw[2160];

#define com_ray_size 20
MPI_Comm com_ray[com_ray_size];
char errstr[256];
char version[8];

#ifdef MPI2
int *array_of_errcodes=NULL;
int array_of_errcodes_size=0;
#endif

MPI_Status status;
int ierr;
void char_func (char *ret, int retl, char *str, int slen, char *str2, int slen2, int *offset);
void the_date(double *since,char* date_str);
void myerror(char *s);
int getptype(long mpitype);
int erroron;

static PyObject *mpiError;

void eh( MPI_Comm *comm, int *err, ... )
{
/*
char string[256];
int len;

 if (*err != MPI_ERR_OTHER) {
 errs++;
 printf( "Unexpected error code\n" );fflush(stdout);
 }
 if (*comm != mycomm) {
 errs++;
 printf( "Unexpected communicator\n" );fflush(stdout);
 }
 calls++;
 */
 ierr=*err;
 /*
 MPI_Error_string(ierr, string,  &len);
 printf( "mpi generated the error %d %s\n",*err,string );fflush(stdout);
 */
 printf( "mpi generated the error %d\n",*err );fflush(stdout);
 return;
}


static PyObject *mpi_get_processor_name(PyObject *self, PyObject *args)
{
/* int MPI_Get_processor_name( char *name, int *resultlen) */
char c_name[MPI_MAX_PROCESSOR_NAME];
int c_len;
        ierr=MPI_Get_processor_name((char *)c_name,&c_len);
      return PyUnicode_FromStringAndSize(c_name,c_len);
}

int getptype(long mpitype) {
	printf("Hello");
    if(mpitype == (long)MPI_INT)    return(PyArray_INT);
    if(mpitype == (long)MPI_FLOAT)  return(PyArray_FLOAT);
    if(mpitype == (long)MPI_DOUBLE) return(PyArray_DOUBLE);
    if(mpitype == (long)MPI_CHAR)   return(PyArray_CHAR);
    if(mpitype == (long)MPI_BYTE) {
    	printf("datatype is MPI_BYTE");
    	return(PyArray_BYTE);/* Added in version for sparx */}
    printf("could not find type input: %ld  available: MPI_FLOAT %ld MPI_INT %ld MPI_DOUBLE %ld MPI_CHAR %ld MPI_BYTE %ld\n",mpitype,(long)MPI_FLOAT,(long)MPI_INT,(long)MPI_DOUBLE,(long)MPI_CHAR,(long)MPI_BYTE);
	printf("This is the place to print out %ld", (long)MPI_BYTE);
	return(PyArray_INT);
}

static PyObject *mpi_test(PyObject *self, PyObject *args)
{
/* int MPI_Test (MPI_Request  *request,int *flag, MPI_Status *status) */
	printf("this routine does not work yet\n");
    return NULL;
}

static PyObject *mpi_wait(PyObject *self, PyObject *args)
{
/* int MPI_Wait (MPI_Request  *request, MPI_Status *status) */
	printf("this routine does not work yet\n");
    return NULL;
}
static PyObject *mpi_isend(PyObject *self, PyObject *args)
{
/* int MPI_Isend( void *buf, int count, MPI_Datatype datatype, int dest, int tag,MPI_Comm comm, MPI_Request *request ) */
	printf("this routine does not work yet\n");
    return NULL;
}
static PyObject *mpi_irecv(PyObject *self, PyObject *args)
{
/* int MPI_Irecv( void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request ) */
	printf("this routine does not work yet\n");
    return NULL;
}

static PyObject *mpi_group_rank(PyObject *self, PyObject *args)
{
/* int MPI_Group_rank ( MPI_Group group, int *rank ) */
long in_group;
int rank;
MPI_Group group;

	if (!PyArg_ParseTuple(args, "l", &in_group))
		return NULL;
	group=(MPI_Group)in_group;
	ierr=MPI_Group_rank (group, &rank );
	return PyLong_FromLong((long)rank);
}

static PyObject *mpi_group_incl(PyObject *self, PyObject *args)
{
/* int MPI_Group_incl ( MPI_Group group, int n, int *ranks, MPI_Group *group_out ) */
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
long in_group;
int *ranks,n;
MPI_Group group,out_group;
PyObject *ranks_obj;
PyArrayObject *array;
char error_message[1024];

	if (!PyArg_ParseTuple(args, "liO", &in_group,&n,&ranks_obj))
        return NULL;
    group=(MPI_Group)in_group;
	array = (PyArrayObject *) PyArray_ContiguousFromObject(ranks_obj, PyArray_INT, 1, 1);
	if (array == NULL)
		return NULL;
	if(array->dimensions[0] < n)
		return NULL;
	ranks=(int*)malloc((size_t) (n*sizeof(int)));
	if (ranks == NULL) {
		sprintf(error_message, "SX_BAD_ALLOC: In mpi_group_incl(), malloc() failed to allocate %d bytes to pointer ranks.\n", n*sizeof(int));
		perror(error_message);
	}
	memcpy((void*)ranks,(void *)(array->data),  (size_t) (n*sizeof(int)));
	ierr=MPI_Group_incl ( (MPI_Group )group, n, ranks, &out_group);
	free(ranks);
	Py_DECREF(array);
	return VERT_FUNC((CAST)out_group);
}

static PyObject *mpi_comm_group(PyObject *self, PyObject *args)
{
/* int MPI_Comm_group ( MPI_Comm comm, MPI_Group *group ) */
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
MPI_Group group;
COM_TYPE comm;
int ierr;

	if (!PyArg_ParseTuple(args, "l", &comm))
        return NULL;
    if((sizeof(MPI_Group) != sizeof(long))  &&  (sizeof(MPI_Group) != sizeof(int)))
    	printf("can not return MPI_Group as long or int sizes %ld %ld %ld",
    	                    sizeof(MPI_Group),sizeof(long),sizeof(int));
    ierr=MPI_Comm_group ( (MPI_Comm) comm, &group );
	return VERT_FUNC((CAST)group);
}

static PyObject *mpi_comm_dup(PyObject *self, PyObject *args)
{
/* int MPI_Comm_dup ( MPI_Comm comm, MPI_Comm *comm_out ) */
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
long tmpcomm;
COM_TYPE  incomm,outcomm;

	if (!PyArg_ParseTuple(args, "l", &tmpcomm))
        return NULL;
    incomm=(COM_TYPE)tmpcomm;
	ierr=MPI_Comm_dup ((MPI_Comm)incomm,(MPI_Comm*)&outcomm );
	return VERT_FUNC((CAST)outcomm);
}


static PyObject *mpi_comm_set_errhandler(PyObject *self, PyObject *args)
{
/* int MPI_Comm_dup ( MPI_Comm comm, MPI_Comm *comm_out ) */
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
long tmpcomm;
COM_TYPE  incomm;
int choice;

	if (!PyArg_ParseTuple(args, "li", &tmpcomm,&choice))
        return NULL;
    incomm=(COM_TYPE)tmpcomm;
#ifdef MPI2
    if(choice == 0)
		ierr= MPI_Comm_set_errhandler( (MPI_Comm)incomm, MPI_ERRORS_ARE_FATAL );
    if(choice == 1)
		ierr= MPI_Comm_set_errhandler( (MPI_Comm)incomm, MPI_ERRORS_RETURN );
    if(choice == 2)
		ierr= MPI_Comm_set_errhandler( (MPI_Comm)incomm, newerr );
#else
    printf("mpi_comm_set_errhandler not supported on this platform\n");
#endif

	return PyLong_FromLong((long)ierr);
}




static PyObject *mpi_comm_create(PyObject *self, PyObject *args)
{
/* int MPI_Comm_create ( MPI_Comm comm, MPI_Group group, MPI_Comm *comm_out ) */
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
COM_TYPE  incomm,outcomm;
long group;

	if (!PyArg_ParseTuple(args, "ll", &incomm,&group))
        return NULL;
	ierr=MPI_Comm_create ((MPI_Comm)incomm,(MPI_Group)group,(MPI_Comm*)&outcomm );
	return VERT_FUNC((CAST)outcomm);
}


static PyObject *mpi_barrier(PyObject *self, PyObject *args)
{
/* int MPI_Barrier ( MPI_Comm comm ) */
COM_TYPE comm;
int ierr;

	if (!PyArg_ParseTuple(args, "l", &comm))
        return NULL;
    ierr=MPI_Barrier((MPI_Comm)comm );
	return PyLong_FromLong((long)ierr);
}
static PyObject *mpi_send(PyObject *self, PyObject *args)
{
/* int MPI_Send( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm ) */
int count,dest,tag;
DATA_TYPE datatype;
COM_TYPE comm;

PyObject *input;
PyArrayObject *array;
char *aptr;
Py_ssize_t ln=0;

	if (!PyArg_ParseTuple(args, "Oiliil", &input, &count,&datatype,&dest,&tag,&comm))
        return NULL;
	if (PyBytes_Check(input)) {
//		printf("b %d %d\n",count,tag);
		PyBytes_AsStringAndSize(input,&aptr,&ln);
//		if (ln!=count) printf("ln %d ct %d\n",ln,count);
		ierr=MPI_Send(aptr,  ln,  (MPI_Datatype)datatype,  dest,  tag,  (MPI_Comm)comm );
		return PyLong_FromLong((long)ierr);
	}
	else {
//		printf("object\n");
		array = (PyArrayObject *) PyArray_ContiguousFromObject(input, getptype(datatype), 0, 3);
		aptr=(char*)(array->data);
		ierr=MPI_Send(aptr,  count,  (MPI_Datatype)datatype,  dest,  tag,  (MPI_Comm)comm );
		Py_DECREF(array);
		return PyLong_FromLong((long)ierr);
	}
	if (array == NULL)
		return NULL;
/*
	n=1;
	for(i=0;i<array->nd;i++)
		n = n* array->dimensions[i];
	if(array->nd == 0)n=1;
	if (n < count)
		return NULL;
*/
}


static PyObject *mpi_recv(PyObject *self, PyObject *args)
{
/* int MPI_Recv( void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status ) */
int count,source,tag;
DATA_TYPE datatype;
COM_TYPE comm;
PyArrayObject *result;
int dimensions[1];
char *aptr;

	if (!PyArg_ParseTuple(args, "iliil", &count,&datatype,&source,&tag,&comm))
        return NULL;
    dimensions[0]=count;
    result = (PyArrayObject *)PyArray_FromDims(1, dimensions, getptype(datatype));
	aptr=(char*)(result->data);
	ierr=MPI_Recv( aptr,  count, (MPI_Datatype)datatype,source,tag, (MPI_Comm)comm, &status );
#ifdef DEBUG
	if(count > 0)
		writeit(result,count,(MPI_Datatype)datatype,"recv");
	else
		dummy("recv");
#endif
  	return PyArray_Return(result);
}

static PyObject *mpi_status(PyObject *self, PyObject *args)
{
/* int MPI_Recv( void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status ) */
PyArrayObject *result;
int dimensions[1],statray[3];

    dimensions[0]=3;
    result = (PyArrayObject *)PyArray_FromDims(1, dimensions, PyArray_INT);
	statray[0]=status.MPI_SOURCE;
	statray[1]=status.MPI_TAG;
	statray[2]=status.MPI_ERROR;
	memcpy((void *)(result->data), (void*)statray, (size_t) (12));
  	return PyArray_Return(result);
}

static PyObject *mpi_error(PyObject *self, PyObject *args)
{
	return PyLong_FromLong((long)ierr);
}


static PyObject * mpi_wtime(PyObject *self, PyObject *args) {

        return( PyFloat_FromDouble(MPI_Wtime()));
}

static PyObject * mpi_wtick(PyObject *self, PyObject *args) {

        return( PyFloat_FromDouble(MPI_Wtick()));
}

static PyObject * mpi_attr_get(PyObject *self, PyObject *args) {
/*
  int MPI_Attr_get ( MPI_Comm comm,int keyval,void *attr_value,int *flag)
Input Parameters
        comm    communicator to which attribute is attached (handle)
        keyval  key value (integer)

Output Parameters
        attr_value      attribute value, unless flag = false
        flag    true if an attribute value was extracted; false if no attribute is associated with the key
*/

        int keyval, *attr_value;
        int flag;
        COM_TYPE comm;
        if (!PyArg_ParseTuple(args, "li", &comm, &keyval))
        return NULL;

/*        printf("mpi_attr_get:  keyval:%d\n",keyval); */

        /* get the keyval for the specified attribute */
        ierr = MPI_Attr_get((MPI_Comm)comm, keyval, &attr_value,&flag);
        if ( !flag ) {
                return NULL;
        }

/*        printf("mpi_attr_get:  attr_val: %d  %d\n",*attr_value,flag); */
        return( PyLong_FromLong((long)*attr_value));
}

#ifdef MPI2
static PyObject *mpi_array_of_errcodes(PyObject *self, PyObject *args)
{
	int dimensions[1];
	if(array_of_errcodes){
	dimensions[0]=array_of_errcodes_size;
	return(PyArray_FromDimsAndData(1,
                        dimensions,
                        PyArray_INT,
                        (char *)array_of_errcodes));
	}
	else {
	dimensions[0]=0;
	return(PyArray_FromDimsAndData(1,
                        dimensions,
                        PyArray_INT,
                        (char *)dimensions));
    }

}

static PyObject *mpi_intercomm_merge(PyObject *self, PyObject *args)
{
/* int MPI_Intercomm_merge ( MPI_Comm comm, int high, MPI_Comm *comm_out ) */
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
COM_TYPE  incomm,outcomm;
int high;

	if (!PyArg_ParseTuple(args, "li", &incomm,&high))
        return NULL;
	ierr=MPI_Intercomm_merge ((MPI_Comm)incomm,high,(MPI_Comm*)&outcomm );
	return VERT_FUNC((CAST)outcomm);
}
static PyObject *mpi_comm_free(PyObject *self, PyObject *args)
{
/* int MPI_Comm_free ( MPI_Comm *comm) */
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
COM_TYPE  incomm;


	if (!PyArg_ParseTuple(args, "l", &incomm))
        return NULL;
	ierr=MPI_Comm_free ((MPI_Comm*)&incomm);
	return VERT_FUNC((CAST)ierr);
}

static PyObject *mpi_comm_spawn(PyObject *self, PyObject *args)
{
/* int MPI_Comm_spawn(char *command, char *argv[], int maxprocs, MPI_Info info,
                  int root, MPI_Comm comm, MPI_Comm *intercomm,
                  int array_of_errcodes[])                    */
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
// int maxprocs,info,root,comm;
int maxprocs,root;
COM_TYPE comm;
MPI_Info info;
MPI_Comm  outcomm;
char *command;
PyObject *input;
int n,len,i;
char **argv;
char error_message[1024];

// if (!PyArg_ParseTuple(args, "sOiiii", &command,&input,&maxprocs,&info,&root,&comm))
if (!PyArg_ParseTuple(args, "sOilii", &command,&input,&maxprocs,&info,&root,&comm))
	return NULL;
	argv=MPI_ARGV_NULL;
	if((input == NULL )|| (input == Py_None )){
		/* printf("is none\n"); */
	}
	if(strncmp("int",input->ob_type->tp_name,3)==0){
		/* printf("is int\n"); */
	}
	if(strncmp("str",input->ob_type->tp_name,3)==0){
		/* printf("is str\n"); */
		argv=(char**)malloc((maxprocs+2)*sizeof(char*));
		if (argv == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_comm_spawn(), malloc() failed to allocate %d bytes to pointer argv.\n", (maxprocs+2)*sizeof(char*));
			perror(error_message);
		}
		
		for(i=0;i<maxprocs+2;i++) {
			argv[i]=(char*)0;
		}
		n=maxprocs;
		if(n > 0){
			for(i=0;i<n;i++) {
				len=strlen(PyUnicode_AsUTF8(input));
				argv[i]=(char*)malloc(len+1);
				if (argv[i] == NULL) {
					sprintf(error_message, "SX_BAD_ALLOC: In mpi_comm_spawn(), malloc() failed to allocate %d bytes to pointer argv[i].\n", len+1);
					perror(error_message);
				}
				argv[i][len]=(char)0;
				strncpy(argv[i],PyUnicode_AsUTF8(input),(size_t)len);
				/* printf("%s\n",argv[i]); */
			}
		}
	}

	if(strncmp("list",input->ob_type->tp_name,4)==0){
//		printf("is list\n");
		argv=(char**)malloc((maxprocs+2)*sizeof(char*));
		if (argv == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_comm_spawn(), malloc() failed to allocate %d bytes to pointer argv.\n", (maxprocs+2)*sizeof(char*));
			perror(error_message);
		}
		
		for(i=0;i<maxprocs+2;i++) {
			argv[i]=(char*)0;
		}
		n=PyList_Size(input);
		if(n > 0){
			for(i=0;i<n;i++) {
				len=strlen(PyUnicode_AsUTF8(PyList_GetItem(input,i)));
				argv[i]=(char*)malloc(len+1);
				if (argv[i] == NULL) {
					sprintf(error_message, "SX_BAD_ALLOC: In mpi_comm_spawn(), malloc() failed to allocate %d bytes to pointer argv[i].\n", len+1);
					perror(error_message);
				}
				argv[i][len]=(char)0;
				strncpy(argv[i],PyUnicode_AsUTF8(PyList_GetItem(input,i)),(size_t)len);
//				printf("%s\n",argv[i]);
			}
		}
	}

	if(array_of_errcodes)free(array_of_errcodes);
	array_of_errcodes=(int*)malloc(maxprocs*sizeof(int));
	if (array_of_errcodes == NULL) {
		sprintf(error_message, "SX_BAD_ALLOC: In mpi_comm_spawn(), malloc() failed to allocate %d bytes to pointer array_of_errcodes.\n", maxprocs*sizeof(int));
		perror(error_message);
	}
	array_of_errcodes_size=maxprocs;

/* int MPI_Comm_spawn(char *command, char *argv[], int maxprocs, MPI_Info info,
                  int root, MPI_Comm comm, MPI_Comm *intercomm,
                  int array_of_errcodes[])                    */
/*	printf("launching %s from %d\n",command,root); */
//	ierr=MPI_Comm_spawn(command,
//	                    argv,
//	                    maxprocs,
//	                    (MPI_Info)info,
//	                    root,
//	                    (MPI_Comm)comm,
//	                    &outcomm,
//	                    array_of_errcodes);
	ierr=MPI_Comm_spawn(command,
	                    argv,
	                    maxprocs,
	                    info,
	                    root,
	                    (MPI_Comm)comm,
	                    &outcomm,
	                    array_of_errcodes);


if(argv != MPI_ARGV_NULL){
		n=maxprocs+2;
		for(i=0;i<n;i++) {
			if(argv[i])free(argv[i]);
		}
		free(argv);
		}

return VERT_FUNC((CAST)outcomm);
}

static PyObject *mpi_comm_get_parent(PyObject *self, PyObject *args)
{
/* int MPI_Comm_get_parent(MPI_Comm *parent)*/
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
COM_TYPE  outcomm;
ierr=MPI_Comm_get_parent((MPI_Comm*)&outcomm);
return VERT_FUNC((CAST)outcomm);
}

static PyObject *mpi_open_port(PyObject *self, PyObject *args)
{
/* int MPI_Open_port ( MPI_Info info, char *port_name) */
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
char  port_name[MPI_MAX_PORT_NAME];
// int info;
MPI_Info info;
	// if (!PyArg_ParseTuple(args, "i", &info))
	if (!PyArg_ParseTuple(args, "l", &info))
		return NULL;
	// ierr=MPI_Open_port((MPI_Info)info,port_name);
	ierr=MPI_Open_port(info,port_name);
	return PyUnicode_FromString(port_name);
}

static PyObject *mpi_close_port(PyObject *self, PyObject *args)
{
/* int MPI_Close_port ( char *port_name) */
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
char  *port_name;
	if (!PyArg_ParseTuple(args, "s", &port_name))
        return NULL;
	ierr=MPI_Close_port(port_name);
	return VERT_FUNC((CAST)ierr);
}

static PyObject *mpi_comm_accept(PyObject *self, PyObject *args)
{
/* int MPI_Comm_accept(char* port_name, MPI_Info info,int root, MPI_Comm comm, MPI_Comm *newcomm)*/
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
COM_TYPE  comm,newcomm;
char* port_name;
// int info,root;
MPI_Info info;
int root;
	// if (!PyArg_ParseTuple(args, "siii", &port_name,&info,&root,&comm))
	if (!PyArg_ParseTuple(args, "slii", &port_name,&info,&root,&comm))
		return NULL;
	// ierr=MPI_Comm_accept(port_name,  (MPI_Info)info, root,  (MPI_Comm)comm,  (MPI_Comm*)&newcomm);
	ierr=MPI_Comm_accept(port_name, info, root, (MPI_Comm)comm, (MPI_Comm*)&newcomm);
	return VERT_FUNC((CAST)newcomm);
}
static PyObject *mpi_comm_connect(PyObject *self, PyObject *args)
{
/* int MPI_Comm_connect(char* port_name, MPI_info info,int root, MPI_Comm comm, MPI_Comm *newcomm)*/
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
COM_TYPE  comm,newcomm;
char* port_name;
// int info,root;
MPI_Info info;
int root;
	// if (!PyArg_ParseTuple(args, "siii", &port_name,&info,&root,&comm))
	if (!PyArg_ParseTuple(args, "slii", &port_name,&info,&root,&comm))
		return NULL;
	// ierr=MPI_Comm_connect(port_name,  (MPI_Info)info, root,  (MPI_Comm)comm,  (MPI_Comm*)&newcomm);
	ierr=MPI_Comm_connect(port_name, info, root, (MPI_Comm)comm, (MPI_Comm*)&newcomm);
	return VERT_FUNC((CAST)newcomm);
}
static PyObject *mpi_comm_disconnect(PyObject *self, PyObject *args)
{
/* int MPI_Comm_disconnect(MPI_Comm *comm)*/
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
COM_TYPE  comm;
	if (!PyArg_ParseTuple(args, "i", &comm))
        return NULL;
	ierr=MPI_Comm_disconnect((MPI_Comm*)&comm);
	return VERT_FUNC((CAST)ierr);
}

#endif

static PyObject *mpi_comm_split(PyObject *self, PyObject *args)
{
/* int MPI_Comm_split ( MPI_Comm comm, int color, int key, MPI_Comm *comm_out ) */
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
int color,key;
COM_TYPE  incomm,outcomm;

	if (!PyArg_ParseTuple(args, "lii", &incomm,&color,&key))
        return NULL;
	ierr=MPI_Comm_split ((MPI_Comm)incomm,color,key,(MPI_Comm*)&outcomm );
	return VERT_FUNC((CAST)outcomm);
}

static PyObject *mpi_probe(PyObject *self, PyObject *args)
{
/* int MPI_Probe( int source, int tag, MPI_Comm comm, MPI_Status *status ) */
int source,tag;
COM_TYPE comm;

	if (!PyArg_ParseTuple(args, "iil", &source,&tag,&comm))
        return NULL;
    ierr=MPI_Probe(source,tag, (MPI_Comm )comm, &status );
	return PyLong_FromLong((long)0);
}

static PyObject *mpi_get_count(PyObject *self, PyObject *args)
{
/* int MPI_Get_count( MPI_Status *status, MPI_Datatype datatype, int *count ) */
DATA_TYPE datatype;
int count;

	if (!PyArg_ParseTuple(args, "l",&datatype))
        return NULL;
    ierr=MPI_Get_count(&status,(MPI_Datatype)datatype,&count);
	return PyLong_FromLong((long)count);
}


static PyObject *mpi_comm_size(PyObject *self, PyObject *args)
{
/* int MPI_Probe( int source, int tag, MPI_Comm comm, MPI_Status *status ) */
COM_TYPE comm;
int numprocs;

	if (!PyArg_ParseTuple(args, "l",&comm))
        return NULL;
	ierr=MPI_Comm_size((MPI_Comm)comm,&numprocs);
	return PyLong_FromLong((long)numprocs);
}

static PyObject *mpi_comm_rank(PyObject *self, PyObject *args)
{
/* int MPI_Probe( int source, int tag, MPI_Comm comm, MPI_Status *status ) */
COM_TYPE comm;
int rank;

	if (!PyArg_ParseTuple(args, "l",&comm))
        return NULL;
	ierr=MPI_Comm_rank((MPI_Comm)comm,&rank);
	return PyLong_FromLong((long)rank);
}


static PyObject *mpi_iprobe(PyObject *self, PyObject *args)
{
/* int MPI_Iprobe( int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status ) */
int source,tag,flag;
COM_TYPE comm;

	if (!PyArg_ParseTuple(args, "iil", &source,&tag,&comm))
        return NULL;
    ierr=MPI_Iprobe(source,tag, (MPI_Comm )comm, &flag,&status );
	return PyLong_FromLong((long)flag);
}

static PyObject * mpi_init(PyObject *self, PyObject *args) {
	PyObject *input;
	int argc,i,n;
	int did_it;
	char **argv;
#ifdef ARG_STR
	char *argstr;
	int *strides;
	int arglen;
#endif
#ifdef SIZE_RANK
	PyArrayObject *result;
	int dimensions[1],data[2];
	char *aptr;
#endif
#ifdef ARG_ARRAY
	PyObject *result;
#endif
	int len;
	argv=NULL;
	erroron=0;
	char error_message[1024];
	
	ierr=MPI_Initialized(&did_it);
	if(!did_it){
		if (!PyArg_ParseTuple(args, "iO", &argc, &input))
			return NULL;
		argv=(char**)malloc((argc+2)*sizeof(char*));
		if (argv == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_init(), malloc() failed to allocate %d bytes to pointer argv.\n", (argc+2)*sizeof(char*));
			perror(error_message);
		}
		n=PyList_Size(input);
		for(i=0;i<n;i++) {
			len=strlen(PyUnicode_AsUTF8(PyList_GetItem(input,i)));
			argv[i]=(char*)malloc(len+1);
			if (argv[i] == NULL) {
				sprintf(error_message, "SX_BAD_ALLOC: In mpi_init(), malloc() failed to allocate %d bytes to pointer argv[i].\n", len+1);
				perror(error_message);
			}
			argv[i][len]=(char)0;
			strncpy(argv[i],PyUnicode_AsUTF8(PyList_GetItem(input,i)),(size_t)len);
			/* printf("%s ",argv[i]); */
		}

		/* printf("\n"); */
		Py_DECREF(input);
#ifdef NULL_INIT
		ierr=MPI_Init(NULL,NULL);
#else
		ierr=MPI_Init(&argc,&argv);
#endif
#ifdef DEBUG
	sprintf(fname,"p%4.4d_%4.4d",myid,(int)getpid());
	ierr=MPI_Get_processor_name((char *)pname,&i);
	debug=fopen(fname,"w");
	fprintf(debug,"%s\n",pname);
#endif

#ifdef MPI2
		MPI_Comm_create_errhandler( eh, &newerr );
#endif

/*		free(argv); */
	}

/*  this returns the command line as a string */
#ifdef ARG_STR
		arglen=0;
		strides=(int*)malloc(argc*sizeof(int));
		if (strides == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_init(), malloc() failed to allocate %d bytes to pointer strides.\n", argc*sizeof(int));
			perror(error_message);
		}
		strides[0]=0;
		for(i=0;i<argc;i++) {
			arglen=arglen+strlen(argv[i])+1;
			strides[i+1]=strides[i]+strlen(argv[i])+1;
		}
		argstr=(char*)malloc(arglen*sizeof(char));
		if (argstr == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_init(), malloc() failed to allocate %d bytes to pointer argstr.\n", arglen*sizeof(char));
			perror(error_message);
		}
		for(i=0;i<argc;i++) {
			for(n=0;n<strlen(argv[i]);n++) {
				argstr[strides[i]+n]=argv[i][n];
			}
			argstr[strides[i]+strlen(argv[i])]=(char)32;
/*
			free(argv[i]);
*/
		}
		return PyUnicode_FromString(argstr);
#endif
#ifdef ARG_ARRAY
		result = PyTuple_New(argc);
		for(i=0;i<argc;i++) {
			PyTuple_SetItem(result,i,PyUnicode_FromString(argv[i]));
		}
/*
for(i=0;i<argc;i++) {
			free(argv[i]);
		}
*/
		return result;
#endif
/*  this returns size and rank */
#ifdef SIZE_RANK
    ierr=MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    ierr=MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	dimensions[0]=2;
	result = (PyArrayObject *)PyArray_FromDims(1, dimensions, PyArray_INT);
	if (result == NULL)
		return NULL;
	data[0]=myid;
	data[1]=numprocs;
	aptr=(char*)&(data);
	for(i=0;i<8;i++)
		result->data[i]=aptr[i];
	if(erroron){ erroron=0; return NULL;}
	return PyArray_Return(result);
#endif
}

static PyObject * mpi_start(PyObject *self, PyObject *args) {
	PyArrayObject *result;
	int argc,did_it,i;
	int dimensions[1],data[2];
	int numprocs,myid;
	char *command,*aptr;
	char **argv;
	char error_message[1024];
	
	erroron=0;
	
	if (!PyArg_ParseTuple(args, "is", &argc, &command))
		return NULL;
	ierr=MPI_Initialized(&did_it);
	if(!did_it){
		/* MPI_Init(0,0); */ /* lam mpi will start with this line
		                        mpich requires us to build a real
		                        command line */
		/* MPI_Init(0,0); */
		argv=(char**)malloc((argc+2)*sizeof(char*));
		if (argv == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_start(), malloc() failed to allocate %d bytes to pointer argv.\n", (argc+2)*sizeof(char*));
			perror(error_message);
		}
		argv[0]=(char*)malloc(128);
		if (argv[0] == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_start(), malloc() failed to allocate %d bytes to pointer argv[0].\n", 128);
			perror(error_message);
		}
		sprintf(argv[0],"dummy");
		strtok(command, " ");
		for(i=0;i<argc-1;i++) {
				argv[i+1]=strtok(NULL, " ");
		}
		printf("calling mpi init from mpi_start\n");
		for (i=0;i<argc;i++)
			printf("%d %d %s\n",i,(int)strlen(argv[i]),argv[i]);
#ifdef NULL_INIT
		ierr=MPI_Init(NULL,NULL);
#else
		ierr=MPI_Init(&argc,&argv);
#endif
	}
	ierr=MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    ierr=MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#ifdef DEBUG
	sprintf(fname,"p%4.4d_%4.4d",myid,(int)getpid());
	ierr=MPI_Get_processor_name((char *)pname,&i);
	debug=fopen(fname,"w");
	fprintf(debug,"%s\n",pname);
#endif
#ifdef MPI2
		MPI_Comm_create_errhandler( eh, &newerr );
#endif


	dimensions[0]=2;
	result = (PyArrayObject *)PyArray_FromDims(1, dimensions, PyArray_INT);
	if (result == NULL)
		return NULL;
	data[0]=myid;
	data[1]=numprocs;
	aptr=(char*)&(data);
	for(i=0;i<8;i++)
		result->data[i]=aptr[i];
	if(erroron){ erroron=0; return NULL;}
	return PyArray_Return(result);
}

static PyObject * mpi_bcast(PyObject *self, PyObject *args) {
/* int MPI_Bcast ( void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm ) */
int count,root;
DATA_TYPE datatype;
COM_TYPE comm;
int myid;
int mysize;
PyArrayObject *result;
PyArrayObject *array;
PyObject *input;
int dimensions[1];
char *aptr;
Py_ssize_t ln=0;
	if (!PyArg_ParseTuple(args, "Oilil", &input, &count,&datatype,&root,&comm))
        return NULL;

	printf("mpi_bcast is called");
	dimensions[0]=count;
	print("count value %d", count);
    result = (PyArrayObject *)PyArray_FromDims(1, dimensions, getptype(datatype));
    printf("result is %i", *result->data);
	aptr=(char*)(result->data);
	printf("aptr values is %i" ,  *aptr);
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);
#ifdef MPI2
    if(myid == root || root == MPI_ROOT) {
#else
    if(myid == root) {
#endif
		if (PyBytes_Check(input)) {
//			printf("bc %d %d\n",count,datatype);
			PyBytes_AsStringAndSize(input,&aptr,&ln);
//			if (ln!=count) printf("lnc %d ct %d\n",ln,count);
			ierr=MPI_Bcast(aptr, ln, (MPI_Datatype)datatype, root, (MPI_Comm)comm);
			return PyLong_FromLong((long)ierr);
		}
		array = (PyArrayObject *) PyArray_ContiguousFromObject(input, getptype(datatype), 0, 3);
		if (array == NULL)
			return NULL;
		ierr=MPI_Type_size((MPI_Datatype)datatype,&mysize);
		memcpy((void *)(result->data), (void*)array->data, (size_t) (mysize*count));
		Py_DECREF(array);
	}
	ierr=MPI_Bcast(aptr,count,(MPI_Datatype)datatype,root,(MPI_Comm)comm);
#ifdef DEBUG
	if(count >0)
		writeit(result,count,(MPI_Datatype)datatype,"bcast");
	else
		dummy("bcast");
#endif
  	return PyArray_Return(result);
}


static PyObject * mpi_scatterv(PyObject *self, PyObject *args) {
/* int MPI_Scatterv(void *sendbuf, int *sendcnts, int *displs, MPI_Datatype sendtype,
                    void *recvbuf, int recvcnt,                MPI_Datatype recvtype,
                    int root, MPI_Comm comm ) */
int root;
COM_TYPE comm;
DATA_TYPE sendtype,recvtype;
PyObject *sendbuf_obj, *sendcnts_obj,*displs_obj;
PyArrayObject *array,*result;
int *sendcnts,*displs,recvcnt;
char *sptr,*rptr;

int numprocs,myid;
int dimensions[1];
char error_message[1024];

	sendcnts=0;
	displs=0;

	array=NULL;
	sptr=NULL;

	if (!PyArg_ParseTuple(args, "OOOlilil", &sendbuf_obj, &sendcnts_obj,&displs_obj,&sendtype,&recvcnt,&recvtype,&root,&comm))
        return NULL;
    /* ! get the number of processors in this comm */
    ierr=MPI_Comm_size((MPI_Comm)comm,&numprocs);
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);


#ifdef MPI2
    if(myid == root || root == MPI_ROOT) {
#else
    if(myid == root) {
#endif
		array = (PyArrayObject *) PyArray_ContiguousFromObject(sendcnts_obj, PyArray_INT, 1, 1);
		if (array == NULL)
			return NULL;
		sendcnts=(int*)malloc((size_t) (sizeof(int)*numprocs));
		if (sendcnts == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_scatterv(), malloc() failed to allocate %d bytes to pointer sendcnts.\n", sizeof(int)*numprocs);
			perror(error_message);
		}
		memcpy((void *)sendcnts, (void*)array->data, (size_t) (sizeof(int)*numprocs));
		Py_DECREF(array);
		array = (PyArrayObject *) PyArray_ContiguousFromObject(displs_obj, PyArray_INT, 1, 1);
		if (array == NULL)
			return NULL;
		displs=(int*)malloc((size_t) (sizeof(int)*numprocs));
		if (displs == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_scatterv(), malloc() failed to allocate %d bytes to pointer displs.\n", sizeof(int)*numprocs);
			perror(error_message);
		}
		memcpy((void *)displs, (void*)array->data, (size_t) (sizeof(int)*numprocs));
		Py_DECREF(array);
		array = (PyArrayObject *) PyArray_ContiguousFromObject(sendbuf_obj, getptype(sendtype), 1, 3);
		if (array == NULL)
			return NULL;
		sptr=(char*)(array->data);
	}

    dimensions[0]=recvcnt;
    result = (PyArrayObject *)PyArray_FromDims(1, dimensions, getptype(recvtype));
    rptr=(char*)(result->data);

	ierr=MPI_Scatterv(sptr, sendcnts, displs, (MPI_Datatype)sendtype,rptr,recvcnt,(MPI_Datatype)recvtype, root, (MPI_Comm )comm );
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);
#ifdef MPI2
    if(myid == root || root == MPI_ROOT) {
#else
    if(myid == root) {
#endif
		Py_DECREF(array);
		free(sendcnts);
		free(displs);
	}
#ifdef DEBUG
	if(recvcnt > 0)
		writeit(result,recvcnt,(MPI_Datatype)recvtype,"scatterv");
	else
		dummy("scatterv");
#endif
  	return PyArray_Return(result);
}

static PyObject * mpi_gatherv(PyObject *self, PyObject *args) {
/*
int MPI_Gatherv ( void *sendbuf, int sendcnt,                MPI_Datatype sendtype,
                  void *recvbuf, int *recvcnts, int *displs, MPI_Datatype recvtype,
                  int root, MPI_Comm comm )
 */
int root;
COM_TYPE comm;
DATA_TYPE sendtype,recvtype;
PyObject *sendbuf_obj, *recvcnts_obj,*displs_obj;
PyArrayObject *array,*result;
int sendcnt,*displs,*recvcnts,rtot,i;
char *sptr,*rptr;
int numprocs,myid;
int dimensions[1];
char error_message[1024];

	displs=0;

	array=NULL;
	sptr=NULL;

	if (!PyArg_ParseTuple(args, "OilOOlil", &sendbuf_obj, &sendcnt,&sendtype,&recvcnts_obj,&displs_obj,&recvtype,&root,&comm))
        return NULL;
    /* ! get the number of processors in this comm */
    ierr=MPI_Comm_size((MPI_Comm)comm,&numprocs);
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);
    rtot=0;
    recvcnts=0;
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);
#ifdef MPI2
    if(myid == root || root == MPI_ROOT) {
#else
    if(myid == root) {
#endif
    /* printf("  get the recv_counts array \n"); */
		array = (PyArrayObject *) PyArray_ContiguousFromObject(recvcnts_obj, PyArray_INT, 1, 1);
		if (array == NULL)
			return NULL;
		recvcnts=(int*)malloc((size_t) (sizeof(int)*numprocs));
		if (recvcnts == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_gatherv(), malloc() failed to allocate %d bytes to pointer recvcnts.\n", sizeof(int)*numprocs);
			perror(error_message);
		}
		memcpy((void *)recvcnts, (void*)array->data, (size_t) (sizeof(int)*numprocs));
		rtot=0;
		for(i=0;i<numprocs;i++)
			rtot=rtot+recvcnts[i];
		Py_DECREF(array);
    /* printf("  get the offset array \n"); */
		array = (PyArrayObject *) PyArray_ContiguousFromObject(displs_obj, PyArray_INT, 1, 1);
		if (array == NULL)
			return NULL;
		displs=(int*)malloc((size_t) (sizeof(int)*numprocs));
		if (displs == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_gatherv(), malloc() failed to allocate %d bytes to pointer displs.\n", sizeof(int)*numprocs);
			perror(error_message);
		}
		memcpy((void *)displs, (void*)array->data, (size_t) (sizeof(int)*numprocs));
		Py_DECREF(array);
	}
	/* printf("  allocate the recvbuf \n"); */
		dimensions[0]=rtot;
		result = (PyArrayObject *)PyArray_FromDims(1, dimensions, getptype(recvtype));
		rptr=(char*)(result->data);
   /* printf("  get sendbuf\n"); */
		array = (PyArrayObject *) PyArray_ContiguousFromObject(sendbuf_obj, getptype(sendtype), 1, 3);
		if (array == NULL)
			return NULL;
		sptr=array->data;


   /* printf("   do the call %d \n",recvcnt); */
	ierr=MPI_Gatherv(sptr, sendcnt, (MPI_Datatype)sendtype,rptr,recvcnts,displs,(MPI_Datatype)recvtype, root, (MPI_Comm )comm );
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);
#ifdef MPI2
    if(myid == root || root == MPI_ROOT) {
#else
    if(myid == root) {
#endif
		free(recvcnts);
		free(displs);
	}
    Py_DECREF(array);
   /* printf("   did the call  %d \n",myid); */
#ifdef DEBUG
	if(recvcnts[myid] > 0)
		writeit(result,recvcnts[myid],(MPI_Datatype)recvtype,"gatherv");
	else
		dummy("gatherv");
#endif
  	return PyArray_Return(result);
}

static PyObject * mpi_gather(PyObject *self, PyObject *args) {
/*
int MPI_Gather ( void *sendbuf, int sendcnt, MPI_Datatype sendtype,
                  void *recvbuf, int recvcnts,
                  MPI_Datatype recvtype,
                  int root, MPI_Comm comm )
 */
int root;
COM_TYPE comm;
DATA_TYPE sendtype,recvtype;
PyObject *sendbuf_obj;
PyArrayObject *array,*result;
int sendcnt,recvcnt,rtot;
char *sptr,*rptr;
int numprocs,myid;
int dimensions[1];

	array=NULL;
	sptr=NULL;

	if (!PyArg_ParseTuple(args, "Oililil", &sendbuf_obj, &sendcnt,&sendtype,&recvcnt,&recvtype,&root,&comm))
        return NULL;
    /* ! get the number of processors in this comm */
    ierr=MPI_Comm_size((MPI_Comm)comm,&numprocs);
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);
    rtot=0;
   /* printf("  get sendbuf\n"); */
	array = (PyArrayObject *) PyArray_ContiguousFromObject(sendbuf_obj, getptype(sendtype), 0, 3);
	if (array == NULL)
		return NULL;
	sptr=array->data;
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);
#ifdef MPI2
    if(myid == root || root == MPI_ROOT) {
#else
    if(myid == root) {
#endif
		rtot=recvcnt*numprocs;
    }
	/* printf("  allocate the recvbuf \n"); */
	dimensions[0]=rtot;
	result = (PyArrayObject *)PyArray_FromDims(1, dimensions, getptype(recvtype));
	rptr=(char*)(result->data);


   /* printf("   do the call %d \n",recvcnt); */
	ierr=MPI_Gather(sptr, sendcnt, (MPI_Datatype)sendtype,rptr,recvcnt,(MPI_Datatype)recvtype, root, (MPI_Comm )comm );
	Py_DECREF(array);
   /* printf("   did the call  %d \n",myid); */
#ifdef DEBUG
	 if(rtot > 0){
	 	writeit(result,recvcnt,(MPI_Datatype)recvtype,"gather");
	 }
	 else {
	 	dummy("gather");
	 }
#endif
  	return PyArray_Return(result);
}

static PyObject * mpi_scatter(PyObject *self, PyObject *args) {
/*
  int MPI_Scatter ( void *sendbuf, int sendcnt, MPI_Datatype sendtype,
                    void *recvbuf, int recvcnt, MPI_Datatype recvtype,
                    int root, MPI_Comm comm )
*/
int root;
COM_TYPE comm;
DATA_TYPE sendtype,recvtype;
PyObject *sendbuf_obj;
PyArrayObject *array,*result;
int sendcnts,recvcnt;
int numprocs,myid;
int dimensions[1];
char *sptr,*rptr;

	sendcnts=0;

	array=NULL;
	sptr=NULL;

	if (!PyArg_ParseTuple(args, "Oililil", &sendbuf_obj, &sendcnts,&sendtype,&recvcnt,&recvtype,&root,&comm))
        return NULL;
    /* ! get the number of processors in this comm */
    ierr=MPI_Comm_size((MPI_Comm)comm,&numprocs);
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);

#ifdef MPI2
    if(myid == root || root == MPI_ROOT) {
#else
    if(myid == root) {
#endif
    /* get sendbuf */
		array = (PyArrayObject *) PyArray_ContiguousFromObject(sendbuf_obj, getptype(sendtype), 1, 3);
		if (array == NULL)
			return NULL;
		    sptr=(char*)(array->data);

    }

    /* allocate the recvbuf */
    dimensions[0]=recvcnt;
    result = (PyArrayObject *)PyArray_FromDims(1, dimensions, getptype(recvtype));
    rptr=(char*)(result->data);

   /*  do the call */
	ierr=MPI_Scatter(sptr, sendcnts, (MPI_Datatype)sendtype,rptr,recvcnt,(MPI_Datatype)recvtype, root, (MPI_Comm )comm );
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);
#ifdef MPI2
    if(myid == root || root == MPI_ROOT) {
#else
    if(myid == root) {
#endif
		Py_DECREF(array);
	}
#ifdef DEBUG
	if(recvcnt > 0)
		writeit(result,recvcnt,(MPI_Datatype)recvtype,"scatter");
	else
		dummy("scatter");
#endif
  	return PyArray_Return(result);
}

static PyObject * mpi_reduce(PyObject *self, PyObject *args) {
/* int MPI_Reduce ( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm ) */
int count,root;
DATA_TYPE datatype;
COM_TYPE comm;
long op;
int myid;
PyArrayObject *result;
PyArrayObject *array;
PyObject *input;
int dimensions[1];
char *sptr,*rptr;

	if (!PyArg_ParseTuple(args, "Oillil", &input, &count,&datatype,&op,&root,&comm))
        return NULL;
    MPI_Comm_rank((MPI_Comm)comm,&myid);
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);
#ifdef MPI2
    if(myid == root || root == MPI_ROOT) {
#else
    if(myid == root) {
#endif
    	dimensions[0]=count;
	}
	else {
		dimensions[0]=0;
	}
    result = (PyArrayObject *)PyArray_FromDims(1, dimensions, getptype(datatype));
	rptr=(char*)(result->data);

	array = (PyArrayObject *) PyArray_ContiguousFromObject(input, getptype(datatype), 0, 3);
	if (array == NULL)
		return NULL;
	sptr=(char*)(array->data);
	ierr=MPI_Reduce(sptr,rptr,count,(MPI_Datatype)datatype,(MPI_Op)op,root,(MPI_Comm)comm);
	Py_DECREF(array);
#ifdef DEBUG
	if(count > 0) {
		writeit(result,count,(MPI_Datatype)datatype,"reduce");
	}
	else
		dummy("reduce");
#endif
  	return PyArray_Return(result);
}


static PyObject * mpi_finalize(PyObject *self, PyObject *args) {
/* int MPI_Finalize() */
	if(erroron){ erroron=0; return NULL;}
    return PyLong_FromLong((long)MPI_Finalize());
}
static PyObject * mpi_alltoall(PyObject *self, PyObject *args) {
/*
   int MPI_Alltoall( void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, int recvcnt,   MPI_Datatype recvtype,
                     MPI_Comm comm )
 */
COM_TYPE comm;
DATA_TYPE sendtype,recvtype;
PyObject *sendbuf_obj;
PyArrayObject *array,*result;
int sendcnts,recvcnt;
int numprocs,myid;
int dimensions[1];
char *sptr,*rptr;
sendcnts=0;

	array=NULL;
	sptr=NULL;

	if (!PyArg_ParseTuple(args, "Oilill", &sendbuf_obj, &sendcnts,&sendtype,&recvcnt,&recvtype,&comm))
        return NULL;
    /* ! get the number of processors in this comm */
    ierr=MPI_Comm_size((MPI_Comm)comm,&numprocs);
    ierr=MPI_Comm_rank((MPI_Comm)comm,&myid);

    /* get sendbuf */
		array = (PyArrayObject *) PyArray_ContiguousFromObject(sendbuf_obj, getptype(sendtype), 1, 3);
		if (array == NULL)
			return NULL;
		    sptr=(char*)(array->data);


    /* allocate the recvbuf */
    dimensions[0]=recvcnt*numprocs;
    result = (PyArrayObject *)PyArray_FromDims(1, dimensions, getptype(recvtype));
    rptr=(char*)(result->data);

   /*  do the call */
	ierr=MPI_Alltoall(sptr, sendcnts, (MPI_Datatype)sendtype,rptr,recvcnt,(MPI_Datatype)recvtype, (MPI_Comm )comm );
	Py_DECREF(array);
#ifdef DEBUG
	if(recvcnt > 0)
		writeit(result,recvcnt,(MPI_Datatype)recvtype,"alltoall");
	else
		dummy("alltoall");
#endif
  	return PyArray_Return(result);
}
static PyObject * mpi_alltoallv(PyObject *self, PyObject *args) {
/*
  int MPI_Alltoallv ( void *sendbuf, int *sendcnts, int *sdispls, MPI_Datatype sendtype,
                      void *recvbuf, int *recvcnts, int *rdispls, MPI_Datatype recvtype,
                      MPI_Comm comm )
*/
COM_TYPE comm;
DATA_TYPE sendtype,recvtype;
PyObject *sendbuf_obj, *recvcnts_obj,*rdispls_obj,*sdispls_obj,*sendcnts_obj;
PyArrayObject *array,*result;
int *sendcnts,*sdispls,*rdispls,*recvcnts,rtot,i;
char *sptr,*rptr;
int numprocs;
int dimensions[1];
#ifdef DEBUG
    int myid;
#endif
char error_message[1024];

	rdispls=0;

	array=NULL;
	sptr=NULL;

	if (!PyArg_ParseTuple(args, "OOOlOOll", &sendbuf_obj, &sendcnts_obj,&sdispls_obj,&sendtype,&recvcnts_obj,&rdispls_obj,&recvtype,&comm))
        return NULL;
    /* ! get the number of processors in this comm */
    ierr=MPI_Comm_size((MPI_Comm)comm,&numprocs);
    rtot=0;
    recvcnts=0;

    /* printf("  get the recvcnts array \n"); */
		array = (PyArrayObject *) PyArray_ContiguousFromObject(recvcnts_obj, PyArray_INT, 1, 1);
		if (array == NULL)
			return NULL;
		recvcnts=(int*)malloc((size_t) (sizeof(int)*numprocs));
		if (recvcnts == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_alltoallv(), malloc() failed to allocate %d bytes to pointer recvcnts.\n", sizeof(int)*numprocs);
			perror(error_message);
		}
		memcpy((void *)recvcnts, (void*)array->data, (size_t) (sizeof(int)*numprocs));
		rtot=0;
		for(i=0;i<numprocs;i++)
			rtot=rtot+recvcnts[i];
		Py_DECREF(array);

    /* printf("  get the recv offset array \n"); */
		array = (PyArrayObject *) PyArray_ContiguousFromObject(rdispls_obj, PyArray_INT, 1, 1);
		if (array == NULL)
			return NULL;
		rdispls=(int*)malloc((size_t) (sizeof(int)*numprocs));
		if (rdispls == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_alltoallv(), malloc() failed to allocate %d bytes to pointer rdispls.\n", sizeof(int)*numprocs);
			perror(error_message);
		}
		memcpy((void *)rdispls, (void*)array->data, (size_t) (sizeof(int)*numprocs));
		Py_DECREF(array);

	/* printf("  allocate the recvbuf \n"); */
		dimensions[0]=rtot;
		result = (PyArrayObject *)PyArray_FromDims(1, dimensions, getptype(recvtype));
		rptr=(char*)(result->data);



    /* printf("  get the sendcnts array \n"); */
		array = (PyArrayObject *) PyArray_ContiguousFromObject(sendcnts_obj, PyArray_INT, 1, 1);
		if (array == NULL)
			return NULL;
		sendcnts=(int*)malloc((size_t) (sizeof(int)*numprocs));
		if (sendcnts == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_alltoallv(), malloc() failed to allocate %d bytes to pointer sendcnts.\n", sizeof(int)*numprocs);
			perror(error_message);
		}
		memcpy((void *)sendcnts, (void*)array->data, (size_t) (sizeof(int)*numprocs));
		Py_DECREF(array);

    /* printf("  get the send offset array \n"); */
		array = (PyArrayObject *) PyArray_ContiguousFromObject(sdispls_obj, PyArray_INT, 1, 1);
		if (array == NULL)
			return NULL;
		sdispls=(int*)malloc((size_t) (sizeof(int)*numprocs));
		if (sdispls == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mpi_alltoallv(), malloc() failed to allocate %d bytes to pointer sdispls.\n", sizeof(int)*numprocs);
			perror(error_message);
		}
		memcpy((void *)sdispls, (void*)array->data, (size_t) (sizeof(int)*numprocs));
		Py_DECREF(array);

   /* printf("  get sendbuf\n"); */
		array = (PyArrayObject *) PyArray_ContiguousFromObject(sendbuf_obj, getptype(sendtype), 1, 3);
		if (array == NULL)
			return NULL;
		sptr=(char*)array->data;

   /* printf("   do the call %d \n"); */
   /*
        MPI_Comm_rank((MPI_Comm)comm,&myid);
   		printf("myid =%d ",myid);
   		for(i=0;i<numprocs;i++)
   			printf("%d ",sendcnts[i]);
   		printf(" | ");
   		for(i=0;i<numprocs;i++)
   			printf("%d ",sdispls[i]);
   		printf(" | ");
   		for(i=0;i<numprocs;i++)
   			printf("%d ",recvcnts[i]);
   		printf(" | ");
   		for(i=0;i<numprocs;i++)
   			printf("%d ",rdispls[i]);
   		printf("\n");
   */
       ierr=MPI_Alltoallv(sptr, sendcnts, sdispls, (MPI_Datatype)sendtype,
                          rptr, recvcnts, rdispls, (MPI_Datatype)recvtype,
                          (MPI_Comm)comm);

		Py_DECREF(array);
		free(recvcnts);
		free(rdispls);
		free(sendcnts);
		free(sdispls);
#ifdef DEBUG
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    if(recvcnts[myid] > 0)
		writeit(result,recvcnts[myid],(MPI_Datatype)recvtype,"alltoallv");
	else
		dummy("alltoallv");
#endif

  	return PyArray_Return(result);
}



static PyObject * mpi_win_allocate_shared(PyObject *self, PyObject *args) {

//	MPI_Win_allocate_shared

//	OMPI_DECLSPEC  int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info,
//											   MPI_Comm comm, void *baseptr, MPI_Win *win);

#ifdef DO_UNSIGED
	#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
/*
	MPI_Init(&argc, &argv);

	int rank_all;
	int rank_sm;
	int size_sm;

	// all communicator
	MPI_Comm comm_sm;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_all);

	// shared memory communicator
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_sm);
	MPI_Comm_rank(comm_sm, &rank_sm);
	MPI_Comm_size(comm_sm, &size_sm);

	std::size_t local_window_count(1000000000);

	char* base_ptr;
	MPI_Win win_sm;
	int disp_unit(sizeof(char));
	MPI_Win_allocate_shared(local_window_count * disp_unit, disp_unit, MPI_INFO_NULL, comm_sm, &base_ptr, &win_sm);

*/
	//NUMPY 1.13 change PyArrayObject *result;
	PyObject *result;
	MPI_Aint size;
	int disp_unit;
	MPI_Info info;
	MPI_Comm comm;
	void *baseptr;
//	MPI_Win *win;
	MPI_Win win;

//	if (!PyArg_ParseTuple(args, "l", &size, &disp_unit, &info, &info, &baseptr, &win))
//		return NULL;
	if (!PyArg_ParseTuple(args, "lill", &size, &disp_unit, &info, &comm))
		return NULL;

//	ierr = MPI_Win_allocate_shared(size, disp_unit, MPI_INFO_NULL, comm, &baseptr, &win);
	ierr = MPI_Win_allocate_shared(size, disp_unit, info, comm, &baseptr, &win);

	result = PyTuple_New(2);
	PyTuple_SetItem(result,0,VERT_FUNC((CAST)win));
	PyTuple_SetItem(result,1,VERT_FUNC((CAST)baseptr));
	return result;

}

static PyObject * mpi_win_shared_query(PyObject *self, PyObject *args) {

//	MPI_Win_shared_query

//	OMPI_DECLSPEC  int MPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr);

#ifdef DO_UNSIGED
	#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif

	//NUMPY 1.13 change PyArrayObject *result;
	PyObject *result;
	MPI_Win win;
	int rank;
	MPI_Aint size;
	int disp_unit;
	void *baseptr;


	if (!PyArg_ParseTuple(args, "li", &win, &rank))
		return NULL;

	ierr = MPI_Win_shared_query(win, rank, &size, &disp_unit, &baseptr);

	result = PyTuple_New(1);
	PyTuple_SetItem(result,0,VERT_FUNC((CAST)baseptr));
	return result;
}

static PyObject * mpi_win_free(PyObject *self, PyObject *args) {

//	MPI_Win_free

//	OMPI_DECLSPEC  int MPI_Win_free(MPI_Win *win);

#ifdef DO_UNSIGED
	#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif

	MPI_Win win;

	if (!PyArg_ParseTuple(args, "l", &win))
		return NULL;

	ierr = MPI_Win_free((MPI_Win*)&win);
	return VERT_FUNC((CAST)ierr);

}





#define PI  3.141592653589793238462643383279502884197
#define EPS 1.0e-16
#define FPMIN 1.0e-30
#define MAXIT 10000
#define XMIN 2.0
#define DOUBLE double

#define MAX( a, b ) ( ( a > b) ? a : b )
#define NRSIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))


#define NUSE1 5
#define NUSE2 5


double bessi0(double x)
{
	double y, ax, ans;
	if ((ax = fabs(x)) < 3.75) {
		y = x / 3.75;
		y *= y;
		ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492+ y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
	}  else {
		y = 3.75 / ax;
		ans = (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1+ y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2+ y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2))))))));
	}
	return ans;
}


DOUBLE chebev(DOUBLE a, DOUBLE b, DOUBLE c[], int m, DOUBLE x)
{
    DOUBLE d = 0.0, dd = 0.0, sv, y, y2;
    int j;

   // if ((x - a)*(x - b) > 0.0)
   //     throw ImageFormatException("x not in range in routine chebev");
    y2 = 2.0 * (y = (2.0 * x - a - b) / (b - a));
    for (j = m - 1;j >= 1;j--)
    {
        sv = d;
        d = y2 * d - dd + c[j];
        dd = sv;
    }
    return y*d - dd + 0.5*c[0];
}


void beschb(DOUBLE x, DOUBLE *gam1, DOUBLE *gam2, DOUBLE *gampl, DOUBLE *gammi)
{
    DOUBLE xx;
    static DOUBLE c1[] =
        {
            -1.142022680371172e0, 6.516511267076e-3,
            3.08709017308e-4, -3.470626964e-6, 6.943764e-9,
            3.6780e-11, -1.36e-13
        };
    static DOUBLE c2[] =
        {
            1.843740587300906e0, -0.076852840844786e0,
            1.271927136655e-3, -4.971736704e-6, -3.3126120e-8,
            2.42310e-10, -1.70e-13, -1.0e-15
        };

    xx = 8.0 * x * x - 1.0;
    *gam1 = chebev(-1.0, 1.0, c1, NUSE1, xx);
    *gam2 = chebev(-1.0, 1.0, c2, NUSE2, xx);
    *gampl = *gam2 - x * (*gam1);
    *gammi = *gam2 + x * (*gam1);
}
#undef NUSE1
#undef NUSE2




void bessjy(DOUBLE x, DOUBLE xnu, DOUBLE *rj, DOUBLE *ry, DOUBLE *rjp, DOUBLE *ryp)
{
    int i, isign, l, nl;
    DOUBLE a, b, br, bi, c, cr, ci, d, del, del1, den, di, dlr, dli, dr, e, f, fact, fact2,
    fact3, ff, gam, gam1, gam2, gammi, gampl, h, p, pimu, pimu2, q, r, rjl,
    rjl1, rjmu, rjp1, rjpl, rjtemp, ry1, rymu, rymup, rytemp, sum, sum1,
    temp, w, x2, xi, xi2, xmu, xmu2;

   // if (x <= 0.0 || xnu < 0.0)
   //     throw ImageFormatException("bad arguments in bessjy");
    nl = (x < XMIN ? (int)(xnu + 0.5) : MAX(0, (int)(xnu - x + 1.5)));
    xmu = xnu - nl;
    xmu2 = xmu * xmu;
    xi = 1.0 / x;
    xi2 = 2.0 * xi;
    w = xi2 / PI;
    isign = 1;
    h = xnu * xi;
    if (h < FPMIN)
        h = FPMIN;
    b = xi2 * xnu;
    d = 0.0;
    c = h;
    for (i = 1;i <= MAXIT;i++)
    {
        b += xi2;
        d = b - d;
        if (fabs(d) < FPMIN)
            d = FPMIN;
        c = b - 1.0 / c;
        if (fabs(c) < FPMIN)
            c = FPMIN;
        d = 1.0 / d;
        del = c * d;
        h = del * h;
        if (d < 0.0)
            isign = -isign;
        if (fabs(del - 1.0) < EPS)
            break;
    }
    //if (i > MAXIT)
    //    throw ImageFormatException("x too large in bessjy; try asymptotic expansion");
    rjl = isign * FPMIN;
    rjpl = h * rjl;
    rjl1 = rjl;
    rjp1 = rjpl;
    fact = xnu * xi;
    for (l = nl;l >= 1;l--)
    {
        rjtemp = fact * rjl + rjpl;
        fact -= xi;
        rjpl = fact * rjtemp - rjl;
        rjl = rjtemp;
    }
    if (rjl == 0.0)
        rjl = EPS;
    f = rjpl / rjl;
    if (x < XMIN)
    {
        x2 = 0.5 * x;
        pimu = PI * xmu;
        fact = (fabs(pimu) < EPS ? 1.0 : pimu / sin(pimu));
        d = -log(x2);
        e = xmu * d;
        fact2 = (fabs(e) < EPS ? 1.0 : sinh(e) / e);
        beschb(xmu, &gam1, &gam2, &gampl, &gammi);
        ff = 2.0 / PI * fact * (gam1 * cosh(e) + gam2 * fact2 * d);
        e = exp(e);
        p = e / (gampl * PI);
        q = 1.0 / (e * PI * gammi);
        pimu2 = 0.5 * pimu;
        fact3 = (fabs(pimu2) < EPS ? 1.0 : sin(pimu2) / pimu2);
        r = PI * pimu2 * fact3 * fact3;
        c = 1.0;
        d = -x2 * x2;
        sum = ff + r * q;
        sum1 = p;
        for (i = 1;i <= MAXIT;i++)
        {
            ff = (i * ff + p + q) / (i * i - xmu2);
            c *= (d / i);
            p /= (i - xmu);
            q /= (i + xmu);
            del = c * (ff + r * q);
            sum += del;
            del1 = c * p - i * del;
            sum1 += del1;
            if (fabs(del) < (1.0 + fabs(sum))*EPS)
                break;
        }
        //if (i > MAXIT)
        //    throw ImageFormatException("bessy series failed to converge");
        rymu = -sum;
        ry1 = -sum1 * xi2;
        rymup = xmu * xi * rymu - ry1;
        rjmu = w / (rymup - f * rymu);
    }
    else
    {
        a = 0.25 - xmu2;
        p = -0.5 * xi;
        q = 1.0;
        br = 2.0 * x;
        bi = 2.0;
        fact = a * xi / (p * p + q * q);
        cr = br + q * fact;
        ci = bi + p * fact;
        den = br * br + bi * bi;
        dr = br / den;
        di = -bi / den;
        dlr = cr * dr - ci * di;
        dli = cr * di + ci * dr;
        temp = p * dlr - q * dli;
        q = p * dli + q * dlr;
        p = temp;
        for (i = 2;i <= MAXIT;i++)
        {
            a += 2 * (i - 1);
            bi += 2.0;
            dr = a * dr + br;
            di = a * di + bi;
            if (fabs(dr) + fabs(di) < FPMIN)
                dr = FPMIN;
            fact = a / (cr * cr + ci * ci);
            cr = br + cr * fact;
            ci = bi - ci * fact;
            if (fabs(cr) + fabs(ci) < FPMIN)
                cr = FPMIN;
            den = dr * dr + di * di;
            dr /= den;
            di /= -den;
            dlr = cr * dr - ci * di;
            dli = cr * di + ci * dr;
            temp = p * dlr - q * dli;
            q = p * dli + q * dlr;
            p = temp;
            if (fabs(dlr - 1.0) + fabs(dli) < EPS)
                break;
        }
        //if (i > MAXIT)
        //    throw ImageFormatException("cf2 failed in bessjy");
        gam = (p - f) / q;
        rjmu = sqrt(w / ((p - f) * gam + q));
        rjmu = NRSIGN(rjmu, rjl);
        rymu = rjmu * gam;
        rymup = rymu * (p + q / gam);
        ry1 = xmu * xi * rymu - rymup;
    }
    fact = rjmu / rjl;
    *rj = rjl1 * fact;
    *rjp = rjp1 * fact;
    for (i = 1;i <= nl;i++)
    {
        rytemp = (xmu + i) * xi2 * ry1 - rymu;
        rymu = ry1;
        ry1 = rytemp;
    }
    *ry = rymu;
    *ryp = xnu * xi * rymu - ry1;
}
#undef EPS
#undef FPMIN
#undef MAXIT
#undef XMIN
#undef NRSIGN

double bessi1_5(double x)
{
    return (x == 0) ? 0 : sqrt(2/(PI*x))*(cosh(x)-sinh(x)/x);
}
double bessj1_5(double x)
{
    double rj, ry, rjp, ryp;
    bessjy(x, 1.5, &rj, &ry, &rjp, &ryp);
    return rj;
}


#define ABS(x) (((x) >= 0) ? (x) : (-(x)))
DOUBLE kfv(DOUBLE w, DOUBLE a, DOUBLE alpha)
{
    DOUBLE sigma = sqrt(ABS(alpha * alpha - (2. * PI * a * w) * (2. * PI * a * w)));

	if (2*PI*a*w > alpha)
		return  pow(2.*PI, 3. / 2.)*pow(a, 3)*bessj1_5(sigma)
				/ (bessi0(alpha)*pow(sigma, 1.5));
	else
		return  pow(2.*PI, 3. / 2.)*pow(a, 3)*bessi1_5(sigma)
				/ (bessi0(alpha)*pow(sigma, 1.5));
}
#undef ABS
#undef PI
#undef DOUBLE

//MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, hostcomm,ierr)

// void mrk_identify_fptr(void *fptr, char *fstr) {
// 	char debug_message[1024];
// 	Dl_info finfo;
// 	int return_code;
// 	return_code = dladdr(fptr, &finfo);
// 	if (!return_code) {
// 		sprintf(debug_message, "MRK_DEBUG: Problem retrieving program information for %x (%s):  %s \n", fptr, fstr, dlerror());
// 	} else {
// 		sprintf(debug_message, "MRK_DEBUG: Address %x (%s) located in %s within %s. \n", fptr, fstr, finfo.dli_fname, finfo.dli_sname);
// 	}
// 	perror(debug_message);
// }

// Function to replace a string with another
// string
char *mrk_replace_substr(const char *a_str, const char *a_old_substr, const char *a_new_substr) {
	char *new_str;
	int i, cnt = 0;
	int new_substr_len = strlen(a_new_substr);
	int old_substr_len = strlen(a_old_substr);
	char error_message[1024];
	
	// Counting the number of times old word
	// occur in the string
	for (i = 0; a_str[i] != '\0'; i++)
	{
		if (strstr(&a_str[i], a_old_substr) == &a_str[i])
		{
			cnt++;
			// Jumping to index after the old word.
			i += old_substr_len - 1;
		}
	}
	
	// Making new string of enough length
	new_str = (char *)malloc(i + cnt * (new_substr_len - old_substr_len) + 1);
	if (new_str == NULL) {
		sprintf(error_message, "SX_BAD_ALLOC: In mrk_replace_substr(), malloc() failed to allocate %d bytes to pointer new_str.\n", i + cnt * (new_substr_len - old_substr_len) + 1);
		perror(error_message);
	}

	i = 0;
	while (*a_str)
	{
		// compare the substring with the new_str
		if (strstr(a_str, a_old_substr) == a_str)
		{
			strcpy(&new_str[i], a_new_substr);
			i += new_substr_len;
			a_str += old_substr_len;
		}
		else
			new_str[i++] = *a_str++;
	}
	
	new_str[i] = '\0';
	return new_str;
}

// Retrieves file basename from path
char* mrk_basename(const char* a_path) {
	char *basename = NULL;
	int path_len = strlen(a_path);
	char error_message[1024];
	
	// Find the index of the last occurrence of path separator ('/' for Linux and '\\' for Win)
	int is_not_found = 1;
	int idx = -1;
	int i = -1;
	for (i = path_len - 1; i >= 0 && idx < 0; --i) {
		if (a_path[i] == '/' || a_path[i] == '\\') {
			idx = i;
		}
	}
	
	if (idx >= 0 && idx < path_len - 1) {
		// There is at least one separator in path (i.e. path contains both directory name and basename)
		// move index to the next one to exclude separator character 
		++idx;
		
		// Allocate enough memory for basename. Remember to make space for '\0'.
		basename = (char *)malloc(strlen(&a_path[idx]) + 1);
		if (basename == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mrk_basename(), malloc() failed to allocate %d bytes to pointer basename.\n", strlen(&a_path[idx]) + 1);
			perror(error_message);
		}
		
		// Copy basename from path
		strcpy(basename, &a_path[idx]);
		
	} else if (idx < 0) {
		// There is no separator. (i.e. path contains only basename)
		// Must copy whole string of path to basename
		// Allocate enough memory for basename. Remember to make space for '\0'.
		basename = (char *)malloc(strlen(a_path) + 1);  // Remember to include '\0'
		if (basename == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mrk_basename(), malloc() failed to allocate %d bytes to pointer basename.\n", strlen(a_path) + 1);
			perror(error_message);
		}
		// Copy basename from path
		strcpy(basename, a_path);
	} else {
		// The last character of the path is separator (i.e. path contains only directory name)
		// Basename must be empty
		// Allocate enough memory for basename. Remember to make space for '\0'.
		basename = (char *)malloc(1);  // Remember to include '\0'
		// There is at least one separator
		if (basename == NULL) {
			sprintf(error_message, "SX_BAD_ALLOC: In mrk_basename(), malloc() failed to allocate %d bytes to pointer basename.\n", 1);
			perror(error_message);
		}
		// Set empty string to basename
		basename[0] = '\0';
	}
	
	return basename;
}

char* mrk_dirname(const char* a_path) {
	char* basname = mrk_basename(a_path);
	char* dirname = mrk_replace_substr(a_path, basname, "");
	free(basname);
	
	return dirname;
}

static PyObject * mpi_iterefa(PyObject *self, PyObject *args) {
	// Toshio Moriya 2018/04/25
	// For error message and debug messages using C way!
	char error_message[1024];
	// char debug_message[1024];
	
//	MPI_Win_free

//	OMPI_DECLSPEC  int MPI_Win_free(MPI_Win *win);

#ifdef DO_UNSIGED
	#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif

	size_t i, kt, j, nit, k;

	MPI_Comm comm;
	//	MPI_Win win;
	float *tvol;
	float *tweight;
	int nxt, nyt, nzt, maxr2, nnxo, myid, color, numprocs, ierr, ir, n_iter;
	
	// Toshio Moriya 2018/05/08
	// This function is assuming nyt == (nxt-1)*2. See below.
	if (!PyArg_ParseTuple(args, "lliiiiiiiili", &tvol, &tweight, &nxt, &nyt, &nzt, &maxr2, &nnxo, &myid, &color, &numprocs, &comm, &n_iter)) {
		sprintf(error_message, "SX_LOGIC_ERROR: In mpi_iterefa(), PyArg_ParseTuple() failed. \n");
		perror(error_message);
		return NULL;
	}
	
	// sprintf(debug_message, "MRK_DEBUG: mpi_iterefa() starting. \n"); if( myid == 0 ) { perror(debug_message); }
	
	// Check the current directory
	if( myid == 0 ) {
		char debug_cwd[1024];
		if (getcwd(debug_cwd, sizeof(debug_cwd)) == NULL) {
			sprintf(error_message, "SX_RUNTIME_ERROR: In mpi_iterefa(), getcwd() failed. \n");
			perror(error_message);
		}
	}
	
	// // Check the library objects of the original fftw functions
	// if( myid == 0 ) {
	// 	sprintf(debug_message, "MRK_DEBUG: Checking the library objects of the original fftw functions. \n");
	// 	perror("MRK_DEBUG: \n");
	// 	perror(debug_message); 
	// 	mrk_identify_fptr(fftw_mpi_init, "fftw_mpi_init");
	// 	mrk_identify_fptr(fftw_mpi_local_size_3d, "fftw_mpi_local_size_3d");
	// 	mrk_identify_fptr(fftw_alloc_real, "fftw_alloc_real");
	// 	mrk_identify_fptr(fftw_mpi_plan_dft_r2c_3d, "fftw_mpi_plan_dft_r2c_3d");
	// 	mrk_identify_fptr(fftw_mpi_plan_dft_c2r_3d, "fftw_mpi_plan_dft_c2r_3d");
	// 	mrk_identify_fptr(fftw_sprint_plan, "fftw_sprint_plan");
	// 	mrk_identify_fptr(fftw_execute, "fftw_execute");
	// 	mrk_identify_fptr(fftw_destroy_plan, "fftw_destroy_plan");
	// }
	
	// Extract file path of the library objects of this function to locate the correct one
	Dl_info finfo;  
	int return_code = dladdr(mpi_iterefa, &finfo);
	if (!return_code) {
		sprintf(error_message, "SX_RUNTIME_ERROR: In mpi_iterefa(), dladdr() could not retrieving a dynamic library object information for %x (%s):  %s \n", mpi_iterefa, "mpi_iterefa", dlerror());
		perror(error_message);
	}
	
	// if( myid == 0 ) {
	// 	sprintf(debug_message, "MRK_DEBUG: Address %x (%s) located in %s within %s. \n", mpi_iterefa, "mpi_iterefa", finfo.dli_fname, finfo.dli_sname);
	// 	perror("MRK_DEBUG: \n");
	// 	perror(debug_message);
	// }
	
	char* dli_fname_correct_dir = mrk_replace_substr(finfo.dli_fname, "python2.7/site-packages/", "");  // remove "python2.7/site-packages/" from directory path
	char* fftw3_lib_dirname = mrk_dirname(dli_fname_correct_dir);
	char* mpi_lib_basename = mrk_basename(dli_fname_correct_dir);
	// if( myid == 0 ) {
	// 	perror("MRK_DEBUG: \n");
	// 	sprintf(debug_message, "MRK_DEBUG: dli_fname_correct_dir := %s \n", dli_fname_correct_dir);
	// 	perror(debug_message);
	// 	sprintf(debug_message, "MRK_DEBUG: fftw3_lib_dirname     :=  %s \n", fftw3_lib_dirname);
	// 	perror(debug_message);
	// 	sprintf(debug_message, "MRK_DEBUG: mpi_lib_basename      :=  %s \n", mpi_lib_basename);
	// 	perror(debug_message);
	// }
	
	// Obtain the function from the correct library objects
	// Declare function pointer variable in proper context
	void (*sxfptr_fftw_mpi_init)(void);  // fftw_mpi_init() -> ibfftw3_mpi.so.3
	int (*sxfptr_fftw_mpi_local_size_3d)(ptrdiff_t, ptrdiff_t, ptrdiff_t, MPI_Comm comm, ptrdiff_t *, ptrdiff_t *);  // fftw_mpi_local_size_3d() -> libfftw3_mpi.so.3
	double *(*sxfptr_fftw_alloc_real)(size_t);  // fftw_alloc_real() -> libfftw3.so.3 
	fftw_plan (*sxfptr_fftw_mpi_plan_dft_r2c_3d)(ptrdiff_t, ptrdiff_t, ptrdiff_t, double *, fftw_complex *, MPI_Comm, unsigned); // fftw_mpi_plan_dft_r2c_3d() -> libfftw3_mpi.so.3
	fftw_plan (*sxfptr_fftw_mpi_plan_dft_c2r_3d)(ptrdiff_t, ptrdiff_t, ptrdiff_t, fftw_complex *, double *, MPI_Comm, unsigned); // fftw_mpi_plan_dft_c2r_3d() -> libfftw3_mpi.so.3
	char *(*sxfptr_fftw_sprint_plan)(const fftw_plan); // fftw_sprint_plan() -> libfftw3.so.3 
	void (*sxfptr_fftw_execute)(const fftw_plan); // fftw_execute() -> libfftw3.so.3
	void (*sxfptr_fftw_destroy_plan)(fftw_plan); // fftw_destroy_plan() -> libfftw3.so.3
	
	
	// load explicitly the correct library I want to use
	char* lib_file_path;
	
	lib_file_path = mrk_replace_substr(dli_fname_correct_dir, mpi_lib_basename, "libfftw3_mpi.so.3");
	// if( myid == 0 ) {
	// 	sprintf(debug_message, "MRK_DEBUG: Generated file_path for libfftw3_mpi.so.3 is %s. \n", lib_file_path);
	// 	perror(debug_message);
	// }
	void *sxhandle_libfftw3_mpi = dlopen(lib_file_path, RTLD_NOW|RTLD_LOCAL);
	if (sxhandle_libfftw3_mpi == NULL) {
		sprintf(error_message, "SX_RUNTIME_ERROR: In mpi_iterefa(), dlopen() failed to obtain handle (sxhandle_libfftw3_mpi) of dynamic library object %s.\n", lib_file_path);
		perror(error_message);
	}
	free(lib_file_path);
	
	lib_file_path = mrk_replace_substr(dli_fname_correct_dir, mpi_lib_basename, "libfftw3.so.3");
	// if( myid == 0 ) {
	// 	sprintf(debug_message, "MRK_DEBUG: Generated file_path for libfftw3_mpi.so.3 is %s. \n", lib_file_path);
	// 	perror(debug_message);
	// }
	void *sxhandle_libfftw3 = dlopen(lib_file_path, RTLD_NOW|RTLD_LOCAL);
	if (sxhandle_libfftw3 == NULL) {
		sprintf(error_message, "SX_RUNTIME_ERROR: In mpi_iterefa(), dlopen() failed to obtain handle (sxhandle_libfftw3) of dynamic library object %s.\n", lib_file_path);
		perror(error_message);
	}
	free(lib_file_path);
	
	free(mpi_lib_basename);
	free(fftw3_lib_dirname);
	free(dli_fname_correct_dir);
	
	// Read the address of the function I want to call later 
	// then assign and cast the function pointers
	sxfptr_fftw_mpi_init = (void (*)(void))dlsym(sxhandle_libfftw3_mpi, "fftw_mpi_init");
	sxfptr_fftw_mpi_local_size_3d = (int (*)(ptrdiff_t, ptrdiff_t, ptrdiff_t, MPI_Comm comm, ptrdiff_t *, ptrdiff_t *))dlsym(sxhandle_libfftw3_mpi, "fftw_mpi_local_size_3d");
	sxfptr_fftw_alloc_real = (double *(*)(size_t))dlsym(sxhandle_libfftw3, "fftw_alloc_real");
	sxfptr_fftw_mpi_plan_dft_r2c_3d = (fftw_plan (*)(ptrdiff_t, ptrdiff_t, ptrdiff_t, double *, fftw_complex *, MPI_Comm, unsigned))dlsym(sxhandle_libfftw3_mpi, "fftw_mpi_plan_dft_r2c_3d");
	sxfptr_fftw_mpi_plan_dft_c2r_3d = (fftw_plan (*)(ptrdiff_t, ptrdiff_t, ptrdiff_t, fftw_complex *, double *, MPI_Comm, unsigned))dlsym(sxhandle_libfftw3_mpi, "fftw_mpi_plan_dft_c2r_3d");
	sxfptr_fftw_sprint_plan = (char *(*)(const fftw_plan))dlsym(sxhandle_libfftw3, "fftw_sprint_plan");
	sxfptr_fftw_execute = (void (*)(const fftw_plan))dlsym(sxhandle_libfftw3, "fftw_execute");
	sxfptr_fftw_destroy_plan = (void (*)(fftw_plan))dlsym(sxhandle_libfftw3, "fftw_destroy_plan");
	
	// // Check the library objects of the explicitly loaded fftw functions
	// if( myid == 0 ) {
	// 	sprintf(debug_message, "MRK_DEBUG: Checking the library objects of the explicitly loaded fftw functions. \n");
	// 	perror("MRK_DEBUG: \n");
	// 	perror(debug_message); 
	// 	mrk_identify_fptr(sxfptr_fftw_mpi_init, "sxfptr_fftw_mpi_init");
	// 	mrk_identify_fptr(sxfptr_fftw_mpi_local_size_3d, "sxfptr_fftw_mpi_local_size_3d");
	// 	mrk_identify_fptr(sxfptr_fftw_alloc_real, "sxfptr_fftw_alloc_real");
	// 	mrk_identify_fptr(sxfptr_fftw_mpi_plan_dft_r2c_3d, "sxfptr_fftw_mpi_plan_dft_r2c_3d");
	// 	mrk_identify_fptr(sxfptr_fftw_mpi_plan_dft_c2r_3d, "sxfptr_fftw_mpi_plan_dft_c2r_3d");
	// 	mrk_identify_fptr(sxfptr_fftw_sprint_plan, "sxfptr_fftw_sprint_plan");
	// 	mrk_identify_fptr(sxfptr_fftw_execute, "sxfptr_fftw_execute");
	// 	mrk_identify_fptr(sxfptr_fftw_destroy_plan, "sxfptr_fftw_destroy_plan");
	// } 
	
	int main_node = 0;

	int nzc = nzt/2;
	int nyc = nyt/2;

	size_t nx_extended = nyt + 1;
	//double *cvv = fftwf_alloc_real(2*size);

	int ncx = nyt/2;

	// Toshio Moriya 2018/05/08
	// Here, this function is assuming nyt == (nxt-1)*2
	float fnorma = 1.0f/(float)(nyt*nyt*nzt);
	
	fftw_plan plan;
	ptrdiff_t alloc_local, local_n0, local_0_start;
//#ifdef False
ierr = MPI_Barrier(comm);
#ifdef False
	printf( "\n  parameters  (nxt, nyt, nzt, maxr2, nnxo, color, myid, numprocs)   %d   %d   %d   %d   %d   %d   %d    %d   %ld",nxt, nyt, nzt, maxr2, nnxo, myid, color, numprocs, comm);
if( myid == 0 ) {
	printf( "\n  vol    %f    %f    %f     %f     %f    %f    %f     %f",tvol[0],tvol[1],tvol[2],tvol[3],tvol[4],tvol[5],tvol[6],tvol[7]);
	printf( "\n  twe    %f    %f    %f     %f\n",tweight[0],tweight[1],tweight[2],tweight[3]);
	double qt = 0.0;
	for (i=0;i<2*nxt*nyt*nzt;++i) qt += tvol[i];
	printf( "\n  tvol sum1     %e",qt);

	qt = 0.0;
	for (i=0;i<nx_extended*nyt*nzt;++i) qt += tvol[i];
	printf( "\n  tvol sum2     %e",qt);
}
#endif
	// fftw_mpi_init();
	sxfptr_fftw_mpi_init();
	//if( myid == 0 ) printf( "\n initialize mpi \n");

ierr = MPI_Barrier(comm);
	
	/* get local data size and allocate   in the order nz ny, nx/2+1 */
	// alloc_local = fftw_mpi_local_size_3d(nzt, nyt, nxt, comm,  &local_n0, &local_0_start);
	alloc_local = sxfptr_fftw_mpi_local_size_3d(nzt, nyt, nxt, comm,  &local_n0, &local_0_start);
	// double *cvv = fftw_alloc_real(2*MAX(alloc_local,1)); //not cleat alloc_local can be zero for dormant cpus
	double *cvv = sxfptr_fftw_alloc_real(2*MAX(alloc_local,1)); //not cleat alloc_local can be zero for dormant cpus
	if (cvv == NULL) {
		sprintf(error_message, "SX_BAD_ALLOC: In mpi_iterefa(), sxfptr_fftw_alloc_real() failed to allocate %d bytes to pointer cvv.\n", MAX(alloc_local,1)*sizeof(float));
		perror(error_message);
	}
	
//>>>printf( "\n  memory allocated    %d    %td    %td     %td\n",myid,alloc_local,local_n0,local_0_start);
ierr = MPI_Barrier(comm);

float startTime = (float)clock()/CLOCKS_PER_SEC;
	// Toshio Moriya 2018/05/08
	// Here, this function is assuming nyt == (nxt-1)*2
	// fftw_plan plan_real_to_complex = fftw_mpi_plan_dft_r2c_3d(nzt, nyt, nyt, cvv, (fftw_complex *)cvv, comm, FFTW_MEASURE);
	fftw_plan plan_real_to_complex = sxfptr_fftw_mpi_plan_dft_r2c_3d(nzt, nyt, nyt, cvv, (fftw_complex *)cvv, comm, FFTW_MEASURE);
	// fftw_plan plan_complex_to_real = fftw_mpi_plan_dft_c2r_3d(nzt, nyt, nyt, (fftw_complex *)cvv, cvv, comm, FFTW_MEASURE);
	fftw_plan plan_complex_to_real = sxfptr_fftw_mpi_plan_dft_c2r_3d(nzt, nyt, nyt, (fftw_complex *)cvv, cvv, comm, FFTW_MEASURE);
	
float endTime = (float)clock()/CLOCKS_PER_SEC -startTime;
///>>>printf( "\n plan ready   time   %f\n",endTime);

//	float *rin = fftwf_alloc_real(alloc_local); //This is buffer for distributed weights
	float *rin = malloc(MAX(alloc_local,1)*sizeof(float)); //This is buffer for distributed weights
	if (rin == NULL) {
		sprintf(error_message, "SX_BAD_ALLOC: In mpi_iterefa(), malloc() failed to allocate %d bytes to pointer rin.\n", MAX(alloc_local,1)*sizeof(float));
		perror(error_message);
	}
	// Distribute tweight -> rin
	int tag1 = 72+color*100;
	int tag2 = 11+color*100;
	if( myid == 0 ) {
		for (ir = 0; ir < numprocs; ++ir) {
			if(ir == 0)  {
				// On main node simply copy
				for (i = 0; i<alloc_local; i++)  rin[i] = tweight[local_0_start+i];//(double)tweight[local_0_start+i];
				//printf( "\n  JOJ    %d    %E    %E",i,tweight[local_0_start+i],rin[i]);}
			} else {
				// receive tlocal_n0, tlocal_0_start, talloc_local
				int itemp[3];
				int tag11 = tag1+ir;
				ierr=MPI_Recv( itemp,  3, (MPI_Datatype)MPI_INT, ir, tag11, comm, &status );
				//printf( "\n GOT IT   %d   %d   %d   %d ",ir,itemp[0],itemp[1],itemp[2]);
				// send part of from tweight[tlocal_0_start*M*(N+1)]
				if( itemp[0] > 0 )  {
					int tag21 = tag2+ir;
					ierr=MPI_Send( &tweight[itemp[1]*nxt*nyt],  itemp[2], (MPI_Datatype)MPI_FLOAT, ir, tag21, comm);
				}
				//printf( "\n SEND  C   %d   %d   %d   %d   %d",ir, itemp[0],itemp[1],itemp[2],itemp[1]*nxt*nyt);
				//printf( "\n  SENDVC   ir   %e    %e    %e     %e     %e    %e    %e     %e",myid,tweight[itemp[1]*nxt*nyt + 0],tweight[itemp[1]*nxt*nyt + 1],tweight[itemp[1]*nxt*nyt + 2],tweight[itemp[1]*nxt*nyt + 3],tweight[itemp[1]*nxt*nyt + 4],tweight[itemp[1]*nxt*nyt + 5],tweight[itemp[1]*nxt*nyt + 6],tweight[itemp[1]*nxt*nyt + 7]);
				//printf( "\n SEND  D  %d\n",ir);
			}
		}
	} else {
		//printf( "\n RECEIVE  %d \n",myid);
		// send local_n0, local_0_start, alloc_local
		int itemp[3] = {local_n0, local_0_start, local_n0*nxt*nyt};
		int tag11 = tag1+myid;
		ierr=MPI_Send( itemp,  3,  (MPI_Datatype)MPI_INT,  main_node,  tag11,  comm );
		//printf( "\n SEND B   %d    %d   %d   %d",myid, itemp[0],itemp[1],itemp[2]);
		if( local_n0 > 0 )  {
			int tag21 = tag2+myid;
			ierr=MPI_Recv( rin, itemp[2], (MPI_Datatype)MPI_FLOAT, main_node, tag21, comm, &status );
		}
		//printf( "\n RECEIVC  %d  %e    %e    %e     %e     %e    %e    %e     %e",myid,rin[0],rin[1],rin[2],rin[3],rin[4],rin[5],rin[6],rin[7]);
	}
	//printf( "\n DONE  %d ",myid);
ierr = MPI_Barrier(comm);
	//printf( "\n MOVING ON   AAA %d \n",myid);
	double *nwe;
	// nwe = fftw_alloc_real(MAX(alloc_local,1));//(double*)malloc( size*sizeof(double) );
	nwe = sxfptr_fftw_alloc_real(MAX(alloc_local,1));
	if (nwe == NULL) {
		sprintf(error_message, "SX_BAD_ALLOC: In mpi_iterefa(), sxfptr_fftw_alloc_real() failed to allocate %d bytes to pointer nwe.\n", MAX(alloc_local,1));
		perror(error_message);
	}
	
	// nwe has to be distributed
	for (kt=0; kt<local_n0; ++kt) {
		int k  = kt+local_0_start;
		int kk = (k>nzc)?(k-nzt):k;
		for (j=0;j<nyt;++j) {
			int jj = (j>nyc)?(j-nyt):j;
			for (i=0;i<nxt;++i) {
				if( (kk*kk+jj*jj+i*i) >= maxr2) {
					rin[i + nxt*(j + kt*nyt)] = 0.0f;
					nwe[i + nxt*(j + kt*nyt)] = 0.0L;
				}  else nwe[i + nxt*(j + kt*nyt)] = 1.0;
			}
		}
	}

	//printf( "\n MOVING ON   BBB %d ",myid);

	int nbel = 5000;
	double *beltab;
	beltab = (double*)malloc( nbel*sizeof(double) );
	if (beltab == NULL) {
		sprintf(error_message, "SX_BAD_ALLOC: In mpi_iterefa(), malloc() failed to allocate %d bytes to pointer beltab.\n", nbel*sizeof(double));
		perror(error_message);
	}
	
	double radius = 1.9*2;
	double alpha = 15.0;
	double normk = kfv(0.0L, radius, alpha);
	for (i=0;i<nbel;++i) {
		double rr = i/(double)(nbel-1)/2.0L;
		beltab[i] = kfv(rr, radius, alpha)/normk;
	}
	
	// size_t niter = 10;
	size_t niter = n_iter;
	for (nit=0; nit<niter; ++nit) {
//if( myid == 0 )  printf( "\n ITERATION  %d ",nit);
		// sprintf(debug_message, "MRK_DEBUG: mpi_iterefa() starting ITERATION %d. \n", nit); if( myid == 0 ) { perror(debug_message); }
		for (k=0; k<local_n0;++k) {
			for (j=0;j<nyt;++j) {
				for (i=0;i<nxt;++i) {
					size_t lad = i + nxt*(j + k*nyt);
					double prd = nwe[lad]*rin[lad]; // rin is weight
					//if(prd > 1.0e44) prd = 1.0e44;//Should not be necessary in DP cout<<"  we have a problem A"<<std::endl;
					lad = 2*i + nx_extended*(j + k*nyt);
					cvv[lad]   = prd;
					cvv[lad+1] = 0.0f;
				}
			}
		}
		
		// fftw_execute(plan_complex_to_real);
		sxfptr_fftw_execute(plan_complex_to_real);
		
		for (kt=0; kt<local_n0;++kt)  {
			int  k = kt+local_0_start;
			float argz = (k<nzc)?(k*k) : (k-nzt)*(k-nzt);
			for (j=0;j<nyt;++j)  {
				float argy = (j<nyc)?(j*j) : (j-nyt)*(j-nyt);
				for (i=0;i<nyt;++i) {
					float argx = (i<ncx)?(i*i) : (i-nyt)*(i-nyt);
					float rr = sqrt(argx + argy + argz)/(nnxo*2.0f);
					int iab = (int)((nbel-1)*rr*2.0f + 0.5f);
					if(iab < nbel) cvv[i + nx_extended*(j + kt*nyt)] *= beltab[iab] * fnorma;  // fnorma is FT normalization factor
					else  cvv[i + nx_extended*(j + kt*nyt)] = 0.0f;
				}
			}
		}
		
//if( myid == 0 ) printf( "\n CVV  %d  %ld  %e    %e    %e     %e     %e    %e    %e     %e\n",myid,nit,cvv[0],cvv[1],cvv[2],cvv[3],cvv[4],cvv[5],cvv[6],cvv[7]);
		// fftw_execute(plan_real_to_complex);
		sxfptr_fftw_execute(plan_real_to_complex);
		
		//for (i=0; i<alloc_local; ++i) nwe[i] /= MAX(1.0e-6, sqrt(cvv[2*i]*cvv[2*i]+cvv[2*i+1]*cvv[2*i+1]));
		double ama = -1.e23;
		double ami = -ama;
		for (k=0; k<local_n0;++k) {
			for (j=0;j<nyt;++j) {
				for (i=0;i<nxt;++i) {
					size_t lan = i + nxt*(j + k*nyt);
					size_t lad = 2*i + nx_extended*(j + k*nyt);
					nwe[lan] /= MAX(1.0e-6, sqrt(cvv[lad]*cvv[lad]+cvv[lad+1]*cvv[lad+1]));
					ama = MAX(ama,nwe[lan]);
					ami = (-1)*MAX(-ami, -nwe[lan]);
				}
			}
		}

//printf( "\n EXTRA  %d  %ld  %e    %e",myid,nit,ama,ami);
//if( myid == 0 ) printf( "\n NWE  %d  %ld  %e    %e    %e     %e     %e    %e    %e     %e\n",myid,nit,nwe[0],nwe[1],nwe[2],nwe[3],nwe[4],nwe[5],nwe[6],nwe[7]);

	}

	// fftw_destroy_plan(plan_real_to_complex);
	// fftw_destroy_plan(plan_complex_to_real);
	sxfptr_fftw_destroy_plan(plan_real_to_complex);
	sxfptr_fftw_destroy_plan(plan_complex_to_real);

	free(beltab);
	free(cvv);
	free(rin);
ierr = MPI_Barrier(comm);
//printf( "\n RACING  %d ",myid);
	//  Bring in nwe block by block and keep multiplying tvol
	if( myid == 0 ) {
		for (ir = 0; ir < numprocs; ++ir) {
			if(ir == 0)  {
				// On main node use the existing block
//printf( "\n  tvol    %f    %f    %f     %f     %f    %f    %f     %f",tvol[0],tvol[1],tvol[2],tvol[3],tvol[4],tvol[5],tvol[6],tvol[7]);
				for (kt=0; kt<local_n0;++kt)  {
					int  k = kt+local_0_start;
					for (j=0;j<nyt;++j)  {
						for (i=0;i<nxt;++i) {
							int lad = 2*i + nx_extended*(j + k*nyt);
							tvol[lad]   *= nwe[i + nxt*(j + kt*nyt)];
							tvol[lad+1] *= nwe[i + nxt*(j + kt*nyt)];
							/*  In DP unnecessary
							double prr = tvol[lad]   * nwe[i + nxt*(j + kt*nyt)];
							double pim = tvol[lad+1] * nwe[i + nxt*(j + kt*nyt)];
							if(prr < 1.0e23 &&  pim < 1.0e23) {
								tvol[lad]   = (float)prr;
								tvol[lad+1] = (float)pim;
							} else {
								double fits = 1.0e23/MAX(prr,pim);
								tvol[lad]   = (float)(prr*fits);
								tvol[lad+1] = (float)(pim*fits);
							}
							*/
						}
					}
				}
				
				free(nwe);
				
				//printf( "\n  JOJ    %d    %E    %E",i,tweight[local_0_start+i],rin[i]);}
			} else {
				// receive tlocal_n0, tlocal_0_start, talloc_local
				int itemp[3];
				int tag11 = tag1+ir;
				ierr=MPI_Recv( itemp,  3, (MPI_Datatype)MPI_INT, ir, tag11, comm, &status );
				//printf( "\n GOT IT   %d   %d   %d   %d \n",ir,itemp[0],itemp[1],itemp[2]);
				// receive part of nwe tweight[tlocal_0_start*M*(N+1)]
				if( itemp[0] > 0 )  {
					double *nwe = (double*)malloc( itemp[2]*sizeof(double) );
					if (nwe == NULL) {
						sprintf(error_message, "SX_BAD_ALLOC: In mpi_iterefa(), malloc() failed to allocate %d byte to pointer nwe.\n", itemp[2]*sizeof(double));
						perror(error_message);
					}
					int tag21 = tag2+ir;
					ierr=MPI_Recv( nwe, itemp[2], (MPI_Datatype)MPI_DOUBLE, ir, tag21, comm, &status );
					//printf( "\n RECEIVED  C   %d  %d\n",ir,ierr);
					for (kt=0; kt<itemp[0];++kt)  {
						int  k = kt+itemp[1];
						for (j=0;j<nyt;++j)  {
							for (i=0;i<nxt;++i) {
								int lad = 2*i + nx_extended*(j + k*nyt);
								tvol[lad]   *= nwe[i + nxt*(j + kt*nyt)];
								tvol[lad+1] *= nwe[i + nxt*(j + kt*nyt)];
								/*  Not needed in DP
								double prr = tvol[lad]  *  nwe[i + nxt*(j + kt*nyt)];
								double pim = tvol[lad+1] * nwe[i + nxt*(j + kt*nyt)];
								if(prr < 1.0e23 &&  pim < 1.0e23) {
									tvol[lad]   = (float)prr;
									tvol[lad+1] = (float)pim;
								} else {
									double fits = 1.0e23/MAX(prr,pim);
									tvol[lad]   = (float)(prr*fits);
									tvol[lad+1] = (float)(pim*fits);
								}
								*/
							}
						}
					}
					free(nwe);
				}
				//printf( "\n SEND  D  %d\n",ir);
			}
		}
	} else {
		//printf( "\n SEND  %d \n",myid);
		// send local_n0, local_0_start, alloc_local
		int itemp[3] = {local_n0, local_0_start, local_n0*nxt*nyt};
		int tag11 = tag1+myid;
		ierr=MPI_Send( itemp,  3,  MPI_INT,  main_node,  tag11,  comm );
		//printf( "\n SEND B   %d    %d   %d   %d\n",myid, itemp[0],itemp[1],itemp[2]);
		// send part of nwe
		if( local_n0 > 0 )  {
			int tag21 = tag2+myid;
			ierr=MPI_Send( nwe, itemp[2], (MPI_Datatype)MPI_DOUBLE, main_node, tag21, comm);
			//printf( "\n SEND  C   %d   %d   %d   %d   %d   %ld\n",myid, ierr, itemp[0],itemp[1],itemp[2],itemp[1]*M*(N+1));
			//for (i = 0; i<alloc_local; i++)  {rin[i] = (double)tin[i];printf( "\n  WOW    %d    %E    %E",i,tin[i],rin[i]);}
		}
		free(nwe);  // Something is always allocated, even on unused CPUs
	}
ierr = MPI_Barrier(comm);
/*
if( myid == 0 ) {
	printf( "\n  vol    %f    %f    %f     %f",tvol[0],tvol[1],tvol[2],tvol[3]);
	printf( "\n  twe    %f    %f    %f     %f\n",tweight[0],tweight[1],tweight[2],tweight[3]);
	double qt = 0.0;
	for (i=0;i<2*size;++i) qt += tvol[i];
	printf( "\n  SECOND  tvol sum     %e",qt);
}
ierr = MPI_Barrier(comm);
*/
//#endif
	//printf( "\n DONE  %d \n",myid);
ierr = MPI_Barrier(comm);

//    printf( "\n generated volume   %td   %td   %d    %d   %E  %E\n",local_n0, local_0_start, alloc_local, myid,  rin[0],rin[1]);

	//NUMPY 1.13 change PyArrayObject *result;
	PyObject *result;
	result = PyTuple_New(1);
	PyTuple_SetItem(result,0,PyUnicode_FromString("1111"));
	
	// sprintf(debug_message, "MRK_DEBUG: mpi_iterefa() is successfully returning! \n"); if( myid == 0 ) { perror(debug_message); }
	
	return result;
//	return PyLong_FromLong((long)ierr);
}

#undef MAX



static PyObject *mpi_comm_split_type(PyObject *self, PyObject *args)
{
/* int MPI_Comm_split ( MPI_Comm comm, int color, int key, MPI_Comm *comm_out ) */
#ifdef DO_UNSIGED
	#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif
	int color,key;
	COM_TYPE  incomm,outcomm;
	MPI_Info info;

	if (!PyArg_ParseTuple(args, "liil", &incomm,&color,&key, &info))
		return NULL;
//	color = MPI_COMM_TYPE_SHARED;

//	ierr=MPI_Comm_split_type((MPI_Comm)incomm, color, key, MPI_INFO_NULL, (MPI_Comm*)&outcomm);
	ierr=MPI_Comm_split_type((MPI_Comm)incomm, color, key, info, (MPI_Comm*)&outcomm);

	return VERT_FUNC((CAST)outcomm);
}



#include <string.h>
#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

static PyObject * pydusa_version(PyObject *self, PyObject *args) {
	int i;
	for (i = 0; i < 2160; i++) {
		cw[i] = (char) 0;
	}
	sprintf(cw, "\n%s\nCompile time: %s  %s\n",STRINGIZE_VALUE_OF(PYDUSA_VERSION), __DATE__, __TIME__);
	return PyUnicode_FromString(cw);
}


static PyObject * copywrite(PyObject *self, PyObject *args) {
int i;
for(i=0;i<2160;i++) {
	cw[i]=(char)0;
}
strncat(cw,"--------------------------------------------------------------\n",80);
strncat(cw,"Copyright (c) 2005 The Regents of the University of California\n",80);
strncat(cw,"All Rights Reserved\n",80);
strncat(cw,"Permission to use, copy, modify and distribute any part of this\n",80);
strncat(cw,"	software for educational, research and non-profit purposes,\n",80);
strncat(cw,"	without fee, and without a written agreement is hereby granted,\n",80);
strncat(cw,"	provided that the above copyright notice, this paragraph and the\n",80);
strncat(cw,"	following three paragraphs appear in all copies.\n",80);
strncat(cw,"Those desiring to incorporate this software into commercial products or\n",80);
strncat(cw,"	use for commercial purposes should contact the Technology\n",80);
strncat(cw,"	Transfer & Intellectual Property Services, University of\n",80);
strncat(cw,"	California, San Diego, 9500 Gilman Drive, Mail Code 0910, La\n",80);
strncat(cw,"	Jolla, CA 92093-0910, Ph: (858) 534-5815, FAX: (858) 534-7345,\n",80);
strncat(cw,"	E-MAIL:invent@ucsd.edu.\n",80);
strncat(cw,"IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY\n",80);
strncat(cw,"	PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR\n",80);
strncat(cw,"	CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF\n",80);
strncat(cw,"	THE USE OF THIS SOFTWARE  EVEN IF THE UNIVERSITY OF\n",80);
strncat(cw,"	CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n",80);
strncat(cw,"THE SOFTWARE PROVIDED HEREIN IS ON AN AS IS BASIS, AND THE\n",80);
strncat(cw,"	UNIVERSITY OF CALIFORNIA HAS NO OBLIGATION TO PROVIDE\n",80);
strncat(cw,"	MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.\n",80);
strncat(cw,"	THE UNIVERSITY OF CALIFORNIA MAKES NO REPRESENTATIONS AND\n",80);
strncat(cw,"	EXTENDS NO WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESS,\n",80);
strncat(cw,"	INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF\n",80);
strncat(cw,"	MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE\n",80);
strncat(cw,"	USE OF THIS SOFTWARE WILL NOT INFRINGE ANY PATENT, TRADEMARK OR\n",80);
strncat(cw,"	OTHER RIGHTS.\n",80);
return PyUnicode_FromString(cw);
}
/*
**** to add ****
mpi_isend
mpi_irecv
mpi_test
mpi_wait
*/


static PyMethodDef mpiMethods[] = {
    {"mpi_win_allocate_shared",	mpi_win_allocate_shared,		METH_VARARGS,     	 mpi_win_allocate_shared__},
    {"mpi_win_shared_query",	mpi_win_shared_query,		METH_VARARGS,     	 mpi_win_shared_query__},
    {"mpi_win_free",	 mpi_win_free,		METH_VARARGS,     	 mpi_win_free__},
    {"mpi_comm_split_type",	mpi_comm_split_type,		METH_VARARGS,     	 mpi_comm_split_type__},
    {"mpi_iterefa",		mpi_iterefa,		METH_VARARGS,     	 mpi_iterefa__},
    {"mpi_alltoall",	mpi_alltoall,		METH_VARARGS,     	 mpi_alltoall__},
    {"mpi_alltoallv",	mpi_alltoallv,      METH_VARARGS,     	 mpi_alltoallv__},
    {"mpi_barrier",		mpi_barrier,		METH_VARARGS,     	 mpi_barrier__},
    {"mpi_bcast",		mpi_bcast,			METH_VARARGS,     	 mpi_bcast__},
    {"mpi_comm_create",	mpi_comm_create,	METH_VARARGS,     	 mpi_comm_create__},
    {"mpi_comm_dup",	mpi_comm_dup,		METH_VARARGS,     	 mpi_comm_dup__},
    {"mpi_comm_group",	mpi_comm_group,		METH_VARARGS,     	 mpi_comm_group__},
    {"mpi_comm_rank",	mpi_comm_rank,		METH_VARARGS,     	 mpi_comm_rank__},
    {"mpi_comm_size",	mpi_comm_size,		METH_VARARGS,     	 mpi_comm_size__},
    {"mpi_comm_split",	mpi_comm_split,		METH_VARARGS,     	 mpi_comm_split__},
    {"mpi_error",		mpi_error,			METH_VARARGS,     	 mpi_error__},
    {"mpi_finalize",	mpi_finalize,		METH_VARARGS,     	 mpi_finalize__},
    {"mpi_gather",		mpi_gather,			METH_VARARGS,     	 mpi_gather__},
    {"mpi_gatherv",		mpi_gatherv,		METH_VARARGS,     	 mpi_gatherv__},
    {"mpi_get_count",	mpi_get_count,		METH_VARARGS,     	 mpi_get_count__},
    {"mpi_group_incl",	mpi_group_incl,		METH_VARARGS,     	 mpi_group_incl__},
    {"mpi_group_rank",	mpi_group_rank,		METH_VARARGS,     	 mpi_group_rank__},
    {"mpi_init",		mpi_init,			METH_VARARGS,     	 mpi_init__},
    {"mpi_iprobe",		mpi_iprobe,			METH_VARARGS,     	 mpi_iprobe__},
    {"mpi_irecv",		mpi_irecv,			METH_VARARGS,     	 mpi_irecv__},
    {"mpi_isend",		mpi_isend,			METH_VARARGS,     	 mpi_isend__},
    {"mpi_probe",		mpi_probe,			METH_VARARGS,     	 mpi_probe__},
    {"mpi_recv",		mpi_recv,			METH_VARARGS,     	 mpi_recv__},
    {"mpi_reduce",		mpi_reduce,			METH_VARARGS,     	 mpi_reduce__},
    {"mpi_scatter",		mpi_scatter,		METH_VARARGS,     	 mpi_scatter__},
    {"mpi_scatterv",	mpi_scatterv,		METH_VARARGS,     	 mpi_scatterv__},
    {"mpi_send",		mpi_send,			METH_VARARGS,     	 mpi_send__},
    {"mpi_start",		mpi_start,			METH_VARARGS,     	 mpi_start__},
    {"mpi_status",		mpi_status,			METH_VARARGS,     	 mpi_status__},
    {"mpi_get_processor_name",		mpi_get_processor_name,			METH_VARARGS,     	 mpi_get_processor_name__},
    {"mpi_test",		mpi_test,			METH_VARARGS,     	 mpi_test__},
    {"mpi_wait",		mpi_wait,			METH_VARARGS,     	 mpi_wait__},
    {"mpi_wtime",		mpi_wtime,			METH_VARARGS,     	 mpi_wtime__},
    {"mpi_wtick",		mpi_wtick,			METH_VARARGS,     	 mpi_wtick__},
    {"mpi_attr_get",	mpi_attr_get,		METH_VARARGS,     	 mpi_attr_get__},
#ifdef DOSLU
    {"par_slu",	        par_slu,		    METH_VARARGS,     	 par_slu__},
    {"boeingheader",    boeingheader,		    METH_VARARGS,     	 par_slu__},
    {"boeingdata",      boeingdata,		    METH_VARARGS,     	 par_slu__},
#endif


#ifdef MPI2
    {"mpi_comm_spawn",		    mpi_comm_spawn,			METH_VARARGS,     	 mpi_comm_spawn__},
    {"mpi_array_of_errcodes",	mpi_array_of_errcodes,	METH_VARARGS,     	 mpi_array_of_errcodes__},
    {"mpi_comm_get_parent",		mpi_comm_get_parent,	METH_VARARGS,     	 mpi_comm_get_parent__},
    {"mpi_comm_free",			mpi_comm_free,			METH_VARARGS,     	 mpi_comm_free__},
    {"mpi_intercomm_merge",		mpi_intercomm_merge,	METH_VARARGS,     	 mpi_intercomm_merge__},
    {"mpi_open_port",			mpi_open_port,			METH_VARARGS,     	 mpi_open_port__},
    {"mpi_close_port",			mpi_close_port,			METH_VARARGS,     	 mpi_close_port__},
    {"mpi_comm_accept",			mpi_comm_accept,		METH_VARARGS,     	 mpi_comm_accept__},
    {"mpi_comm_connect",		mpi_comm_connect,		METH_VARARGS,     	 mpi_comm_connect__},
    {"mpi_comm_disconnect",		mpi_comm_disconnect,	METH_VARARGS,     	 mpi_comm_disconnect__},
    {"mpi_comm_set_errhandler",	mpi_comm_set_errhandler,METH_VARARGS,     	 mpi_comm_set_errhandler__},

#endif
    {"copywrite",		copywrite,			METH_VARARGS,     	COPYWRITE_STR__},
    {"pydusa_version",		pydusa_version,			METH_VARARGS,     	pydusa_version__},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/*
*/

static struct PyModuleDef mpiModuleDef = {
        PyModuleDef_HEAD_INIT,
        "mpi",
        NULL,
        -1,
        mpiMethods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit_mpi(void)
{
#ifdef DO_UNSIGED
#define CAST unsigned long
#define VERT_FUNC PyLong_FromUnsignedLong
#else
#define CAST long
#define VERT_FUNC PyLong_FromLong
#endif

	PyObject *m, *d;
    PyObject *tmp;
	import_array();
	m=PyModule_Create(&mpiModuleDef);
	mpiError = PyErr_NewException("mpi.error", NULL, NULL);
    Py_INCREF(mpiError);
    PyModule_AddObject(m, "error", mpiError);
    d = PyModule_GetDict(m);
    tmp = PyUnicode_FromString(VERSION);
    PyDict_SetItemString(d,   "VERSION", tmp);  Py_DECREF(tmp);
    tmp = PyLong_FromLong((long)MPI_VERSION);
    PyDict_SetItemString(d,   "MPI_VERSION", tmp);  Py_DECREF(tmp);
    tmp = PyLong_FromLong((long)MPI_SUBVERSION);
    PyDict_SetItemString(d,   "MPI_SUBVERSION", tmp);  Py_DECREF(tmp);
    tmp = PyUnicode_FromString(COPYWRITE);
    PyDict_SetItemString(d,   "COPYWRITE", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_CHAR);
    PyDict_SetItemString(d,   "MPI_CHAR", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_BYTE);
    PyDict_SetItemString(d,   "MPI_BYTE", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_SHORT);
    PyDict_SetItemString(d,   "MPI_SHORT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_INT);
    PyDict_SetItemString(d,   "MPI_INT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_LONG);
    PyDict_SetItemString(d,   "MPI_LONG", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_FLOAT);
    PyDict_SetItemString(d,   "MPI_FLOAT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_DOUBLE);
    PyDict_SetItemString(d,   "MPI_DOUBLE", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_UNSIGNED_CHAR);
    PyDict_SetItemString(d,   "MPI_UNSIGNED_CHAR", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_UNSIGNED_SHORT);
    PyDict_SetItemString(d,   "MPI_UNSIGNED_SHORT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_UNSIGNED);
    PyDict_SetItemString(d,   "MPI_UNSIGNED", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_UNSIGNED_LONG);
    PyDict_SetItemString(d,   "MPI_UNSIGNED_LONG", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_LONG_DOUBLE);
    PyDict_SetItemString(d,   "MPI_LONG_DOUBLE", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_FLOAT_INT);
    PyDict_SetItemString(d,   "MPI_FLOAT_INT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_LONG_INT);
    PyDict_SetItemString(d,   "MPI_LONG_INT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_DOUBLE_INT);
    PyDict_SetItemString(d,   "MPI_DOUBLE_INT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_SHORT_INT);
    PyDict_SetItemString(d,   "MPI_SHORT_INT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_2INT);
    PyDict_SetItemString(d,   "MPI_2INT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_LONG_DOUBLE_INT);
    PyDict_SetItemString(d,   "MPI_LONG_DOUBLE_INT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_LONG_LONG_INT);
    PyDict_SetItemString(d,   "MPI_LONG_LONG_INT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_PACKED);
    PyDict_SetItemString(d,   "MPI_PACKED", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_Pack);
    PyDict_SetItemString(d,   "MPI_Pack", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_Unpack);
    PyDict_SetItemString(d,   "MPI_Unpack", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_UB);
    PyDict_SetItemString(d,   "MPI_UB", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_LB);
    PyDict_SetItemString(d,   "MPI_LB", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_MAX);
    PyDict_SetItemString(d,   "MPI_MAX", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_MIN);
    PyDict_SetItemString(d,   "MPI_MIN", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_SUM);
    PyDict_SetItemString(d,   "MPI_SUM", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_PROD);
    PyDict_SetItemString(d,   "MPI_PROD", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_LAND);
    PyDict_SetItemString(d,   "MPI_LAND", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_BAND);
    PyDict_SetItemString(d,   "MPI_BAND", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_LOR);
    PyDict_SetItemString(d,   "MPI_LOR", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_BOR);
    PyDict_SetItemString(d,   "MPI_BOR", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_LXOR);
    PyDict_SetItemString(d,   "MPI_LXOR", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_BXOR);
    PyDict_SetItemString(d,   "MPI_BXOR", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_MINLOC);
    PyDict_SetItemString(d,   "MPI_MINLOC", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_MAXLOC);
    PyDict_SetItemString(d,   "MPI_MAXLOC", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_MAXLOC);
    PyDict_SetItemString(d,   "MPI_MAXLOC", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_COMM_NULL);
    PyDict_SetItemString(d,   "MPI_COMM_NULL", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_OP_NULL);
    PyDict_SetItemString(d,   "MPI_OP_NULL", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_GROUP_NULL);
    PyDict_SetItemString(d,   "MPI_GROUP_NULL", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_DATATYPE_NULL);
    PyDict_SetItemString(d,   "MPI_DATATYPE_NULL", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_REQUEST_NULL);
    PyDict_SetItemString(d,   "MPI_REQUEST_NULL", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_ERRHANDLER_NULL);
    PyDict_SetItemString(d,   "MPI_ERRHANDLER_NULL", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_MAX_PROCESSOR_NAME);
    PyDict_SetItemString(d,   "MPI_MAX_PROCESSOR_NAME", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_MAX_ERROR_STRING);
    PyDict_SetItemString(d,   "MPI_MAX_ERROR_STRING", tmp);  Py_DECREF(tmp);
	tmp = PyLong_FromLong((long)MPI_UNDEFINED);
    PyDict_SetItemString(d,   "MPI_UNDEFINED", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_KEYVAL_INVALID);
    PyDict_SetItemString(d,   "MPI_KEYVAL_INVALID", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_BSEND_OVERHEAD);
    PyDict_SetItemString(d,   "MPI_BSEND_OVERHEAD", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_PROC_NULL);
    PyDict_SetItemString(d,   "MPI_PROC_NULL", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_ANY_SOURCE);
    PyDict_SetItemString(d,   "MPI_ANY_SOURCE", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_ANY_TAG);
    PyDict_SetItemString(d,   "MPI_ANY_TAG", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_BOTTOM);
    PyDict_SetItemString(d,   "MPI_BOTTOM", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_COMM_WORLD);
    PyDict_SetItemString(d,   "MPI_COMM_WORLD", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_TAG_UB);
    PyDict_SetItemString(d,   "MPI_TAG_UB", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_HOST);
    PyDict_SetItemString(d,   "MPI_HOST", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_IO);
    PyDict_SetItemString(d,   "MPI_IO", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_WTIME_IS_GLOBAL);
    PyDict_SetItemString(d,   "MPI_WTIME_IS_GLOBAL", tmp);  Py_DECREF(tmp);

    tmp = PyUnicode_FromString(LIBRARY);
    PyDict_SetItemString(d,   "ARRAY_LIBRARY", tmp);  Py_DECREF(tmp);

    tmp = PyUnicode_FromString(DATE_DOC);
    PyDict_SetItemString(d,   "DATE_DOC", tmp);  Py_DECREF(tmp);
    tmp = PyUnicode_FromString(DATE_SRC);
    PyDict_SetItemString(d,   "DATE_SRC", tmp);  Py_DECREF(tmp);

#ifdef MPI2
    tmp = VERT_FUNC((CAST)MPI_UNIVERSE_SIZE);
    PyDict_SetItemString(d,   "MPI_UNIVERSE_SIZE", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_ROOT);
    PyDict_SetItemString(d,   "MPI_ROOT", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_ARGV_NULL);
    PyDict_SetItemString(d,   "MPI_ARGV_NULL", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_INFO_NULL);
    PyDict_SetItemString(d,   "MPI_INFO_NULL", tmp);  Py_DECREF(tmp);
    tmp = VERT_FUNC((CAST)MPI_COMM_TYPE_SHARED);
    PyDict_SetItemString(d,   "MPI_COMM_TYPE_SHARED", tmp);  Py_DECREF(tmp);
#endif



return m;


}
void myerror(char *s) {
	erroron=1;
	PyErr_SetString(mpiError,s);
}


#ifdef DOSLU
#include "./solvers.c"
#endif
