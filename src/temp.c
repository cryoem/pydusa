#include <stdio.h>

int main(int argc, char** argv) {

int myid = 0;
int kt, k,  local_n0, local_0_start, nyt, i,j, lad, nx_extended;

if( myid == 0 ) {
for (kt=0; kt<local_n0;++kt) {
k = kt+local_0_start;
for (j=0;j<nyt;++j)  {
//				for (i=0;i<nxt;++i) {
//					lad = 2*i + nx_extended*(j + k*nyt);
printf("\n  ADDRESSING    %d    %d    %d     %d     %d     %d", i, j, k, lad, nx_extended, local_0_start);
//				}
}
}
}
    printf("\nHello, World!\n");
}



