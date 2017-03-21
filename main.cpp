    //compile with mpic++ main.cpp
// run with mpirun -np 4 ./a.out


#include <iostream>
#include <mpi.h>


int main(int argc, char** argv) {
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

    std::size_t local_window_count(10);
    MPI_Aint size;

    char* base_ptr;
    MPI_Win win_sm;
    int disp_unit(sizeof(char));

    if (rank_sm != 0) local_window_count = 0;

    MPI_Win_allocate_shared(local_window_count * disp_unit, disp_unit, MPI_INFO_NULL, comm_sm, &base_ptr, &win_sm);

    MPI_Win_shared_query(win_sm, 0, &size, &disp_unit, &base_ptr);

    local_window_count = 10;

    // write
    char buffer;
    if (rank_sm == 0) {
        buffer = 'A';
    }
    else if (rank_sm == 1) {
        buffer = 'C';
    }
    else if (rank_sm == 2) {
        buffer = 'G';
    }
    else {
        buffer = 'T';
    }

    MPI_Win_fence(0, win_sm);

    for (std::size_t it = 0; it < local_window_count; it++) {
        if (rank_sm == 1) base_ptr[it] = buffer;
    }

    MPI_Win_fence(0, win_sm);


    for (std::size_t it = 0; it < local_window_count; it++) {
        std::cout << rank_sm << " ---- " << base_ptr[it] << std::endl;
    }

    MPI_Finalize();

    return 0;
}
