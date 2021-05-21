// Use GTX 980 machines
// To compile use
// /usr/local/cuda-11.2/bin/nvcc -arch=compute_52 -o multiple_shared.out multiple_shared.cu
// To run use
// ./multiple_shared.out
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <assert.h>

#define ITER 20
#define BILLION 1000000000
#define BLOCK_SIZE 32 // max is 32
#define MATRIX_SIZE 64

void write_output(int *mat, int m, int n, FILE *res)
{
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      fprintf(res, "%d ", mat[i*n+j]);
    }
    fprintf(res, "\n");
  }
  for (int i = 0; i < 3; i++)
  {
    fprintf(res, "\n");
  }
}

__global__ void update_matrix(int *current, int *future, int m, int n)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x; //bigger
  int y = blockIdx.y * blockDim.y + threadIdx.y; //bigger
  int x_sh = x%BLOCK_SIZE;
  int y_sh = y%BLOCK_SIZE;
  
  __shared__ int curr_shared[BLOCK_SIZE*BLOCK_SIZE];
  
  curr_shared[x_sh*BLOCK_SIZE+y_sh] = current[x*m+y];

  __syncthreads();

  if (x >= m-1 || y >= n-1) future[x*m + y] = 0;
  else if (x==0 || y == 0) future[x*m + y] = 0;
  else {
    int aliveN = 0;
    for (int i = -1; i <= 1; i++)
    {
      for (int j = -1; j <= 1; j++)
      {
        //   if((x_sh + i)*BLOCK_SIZE + y_sh + j < )
        if ((x_sh + i) < 0 || (x_sh + i) >= BLOCK_SIZE || y_sh + j < 0 || (y_sh + j) >= BLOCK_SIZE ) {
            aliveN += current[(x + i)*MATRIX_SIZE + y + j];
        }
        else aliveN += curr_shared[(x_sh + i)*BLOCK_SIZE + y_sh + j];
      }
    }
    aliveN -= curr_shared[x_sh*BLOCK_SIZE + y_sh];
    //if lonely it dies
    if (aliveN < 2 && curr_shared[x_sh*BLOCK_SIZE + y_sh] == 1) {
      future[x*m + y] = 0;
    }
    //if overpopulated it dies
    else if (aliveN > 3 && curr_shared[x_sh*BLOCK_SIZE + y_sh] == 1)
    {
      future[x*m + y] = 0;
    }
    // if repopulated it revives
    else if (aliveN == 3 && curr_shared[x_sh*BLOCK_SIZE + y_sh] == 0) {
      future[x*m + y] = 1;
    }
    // else copy current to future
    else
    {
      future[x*m + y] = curr_shared[x_sh*BLOCK_SIZE + y_sh];
    }
  }
  __syncthreads();
}

int main()
{
  cudaError_t cudaStat = cudaSuccess;
  int i, j;
  int m, n;
  int *dev_even, *dev_odd;

  FILE *res;
  res = fopen("output_multiple_shared.txt", "w");

  m = n = MATRIX_SIZE;

  srand(0);
  int *even = (int*) calloc(m * n *sizeof(int), sizeof(int));
  for (i = 1; i < m - 1; i++)
  {
    for (j = 1; j < n - 1; j++)
    {
      even[i*m+j] = rand() % 2;
    }
  }
  int *odd = (int *) calloc(m*n*sizeof(int), sizeof(int));
  write_output(even, m, n, res);

  dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 Grid(MATRIX_SIZE/BLOCK_SIZE, MATRIX_SIZE/BLOCK_SIZE);

  cudaMalloc((void **) &dev_even, m*n*sizeof(int));
  cudaMalloc((void **) &dev_odd, m*n*sizeof(int));
  for (int iter = 0; iter < ITER; iter++)
  {
    if (iter % 2 == 0)
    {
      cudaStat = cudaMemcpy(dev_even,even,m*n*sizeof(int),cudaMemcpyHostToDevice);
      assert(cudaStat == cudaSuccess);
      cudaStat = cudaMemcpy(dev_odd,odd,m*n*sizeof(int),cudaMemcpyHostToDevice);
      assert(cudaStat == cudaSuccess);
      update_matrix<<<Grid, Block>>>(dev_even,dev_odd,m,n);
      cudaStat = cudaMemcpy(odd,dev_odd,m*n*sizeof(int),cudaMemcpyDeviceToHost);
      assert(cudaStat == cudaSuccess);
      write_output(odd, m, n, res);
    }
    if (iter % 2 == 1)
    {
      cudaStat = cudaMemcpy(dev_even,even,m*n*sizeof(int),cudaMemcpyHostToDevice);
      assert(cudaStat == cudaSuccess);
      cudaStat = cudaMemcpy(dev_odd,odd,m*n*sizeof(int),cudaMemcpyHostToDevice);
      assert(cudaStat == cudaSuccess);
      update_matrix<<<Grid, Block>>>(dev_odd,dev_even,m,n);
      cudaStat = cudaMemcpy(even,dev_even,m*n*sizeof(int),cudaMemcpyDeviceToHost);
      assert(cudaStat == cudaSuccess);
      write_output(even, m, n, res);
    }
  }
  fclose(res);
  free(even);
  free(odd);
  cudaFree(dev_even);
  cudaFree(dev_odd);
}
