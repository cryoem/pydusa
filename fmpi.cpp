#include <fftw3-mpi.h>
#include <time.h>
//   mpic++ -lfftw3_mpi -lfftw3 -lm fmpi.cpp
 int main(int argc, char **argv)
 {
//ptrdiff_t lst = 259;
//ptrdiff_t lst = 256;
ptrdiff_t lst = 256;

//	std::cin.getline(argv[1], sizeof(argv[1])) >> lst;
	lst = atoi(argv[1]);
	//const ptrdiff_t L = 259, M = 259, N = 259;
	const ptrdiff_t L = lst, M = lst, N = lst;
	fftw_plan plan;
	double *rin;
	fftw_complex *cout;
	ptrdiff_t alloc_local, local_n0, local_0_start, i, j, k;

//	 std::cout << "\nAAAAAAAAA\n";
//
//	 exit(0);


	MPI_Init(&argc, &argv);
	float startTime, endTime;



	int rank_all;
	int rank_sm;
	int size_sm;

	// all communicator
	MPI_Comm comm_sm;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_all);

#ifdef False

	double* Bin = fftw_alloc_real(2 * L*M*(N/2+1));
	fftw_complex* Bout = fftw_alloc_complex(L*M*(N/2+1));
	fftw_plan plan_real_to_complex = fftw_plan_dft_r2c_3d(L, M, N, Bin, Bout, FFTW_ESTIMATE);



	/* initialize rin to some function my_func(x,y,z) */
	for (i = 0; i < L; ++i)
		for (j = 0; j < M; ++j)
			for (k = 0; k < N; ++k)
				Bin[(i*M + j) * (2*(N/2+1)) + k] = float(i+j+k);
 
startTime = (float)clock()/CLOCKS_PER_SEC;
for    (i = 0; i < 10; ++i)   {
std::cout<<"  loop  "<<i<<std::endl;
fftw_execute(plan_real_to_complex);
}  
endTime = (float)clock()/CLOCKS_PER_SEC;
std::cout<<"  time  "<<endTime - startTime<<std::endl;



 fftw_destroy_plan(plan_real_to_complex);
#endif

//#ifdef False
if( rank_all == 0 ) std::cout<<"  initialize mpi "<<std::endl;

		 fftw_mpi_init();

		 /* get local data size and allocate */
		 alloc_local = fftw_mpi_local_size_3d(L, M, N/2+1, MPI_COMM_WORLD,  &local_n0, &local_0_start);
		 rin = fftw_alloc_real(2 * alloc_local);
		 cout = fftw_alloc_complex(alloc_local);
if( rank_all == 0 ) std::cout<<"  memory allocated "<<std::endl;

		startTime = (float)clock()/CLOCKS_PER_SEC;
		/* create plan for out-of-place r2c DFT */
		plan = fftw_mpi_plan_dft_r2c_3d(L, M, N, rin, cout, MPI_COMM_WORLD, FFTW_MEASURE);
//	std::cout << "\nAAAAAAAAA\n";
//	MPI_Finalize();
//	exit(0);
		endTime = (float)clock()/CLOCKS_PER_SEC;
		std::cout<<"  plan ready time  "<<endTime - startTime<<std::endl;
   
	 /* initialize rin to some function my_func(x,y,z) */
	 for (i = 0; i < local_n0; ++i)
		for (j = 0; j < M; ++j)
			for (k = 0; k < N; ++k)
				rin[(i*M + j) * (2*(N/2+1)) + k] =float(local_0_start+i+j+k);// my_func(local_0_start+i, j, k);
std::cout<<"  generated volume "<<local_n0<<"  "<<rank_all<<std::endl;
         /* compute transforms as many times as desired */

startTime = (float)clock()/CLOCKS_PER_SEC;

for    (i = 0; i < 10; ++i)   {
	if( rank_all == 0 ) std::cout<<"  loop  "<<i<<std::endl;
	fftw_execute(plan);
}  

endTime = (float)clock()/CLOCKS_PER_SEC;
std::cout<<"  time  "<<endTime - startTime<<std::endl;


         fftw_destroy_plan(plan);
//#endif  
         MPI_Finalize();
     return 0;
}
