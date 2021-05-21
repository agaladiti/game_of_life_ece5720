// Use GTX 980 machines
// To compile use
// /usr/local/cuda-11.2/bin/nvcc -arch=compute_52 -o multiple_global.out multiple_global.cu
// To run use
// ./multiple_global.out
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <assert.h>

#define ITER 20
#define BILLION 1000000000
#define BLOCK_SIZE 32
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
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= m-1 || y >= n-1) future[x*m + y] = 0;
  else if (x==0 || y == 0) future[x*m + y] = 0;
  else{
    int aliveN = 0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
        aliveN += current[(x + i)*m + y + j];
        }
    }
    aliveN -= current[x*m + y];
    __syncthreads();
    //if lonely it dies
    if (aliveN < 2 && current[x*m + y] == 1)
        future[x*m + y] = 0;
    //if overpopulated it dies
    else if (aliveN > 3 && current[x*m + y] == 1)
    {
        future[x*m + y] = 0;
    }
    // if repopulated it revives
    else if (aliveN == 3 && current[x*m + y] == 0)
        future[x*m + y] = 1;
    // else copy current to future
    else
    {
        future[x*m + y] = current[x*m + y];
    }
  }
  __syncthreads();
}

int main()
{
  cudaError_t cudaStat = cudaSuccess;
  cudaEvent_t cstart, cstop;
  cudaEventCreate(&cstart);
  cudaEventCreate(&cstop);
  int i, j;
  int m, n;
  int *dev_even, *dev_odd;
  float time;

  FILE *res;
  res = fopen("output_multiple_global.txt", "w");

  m = n = MATRIX_SIZE;

  srand(0);
  int *even = (int*) calloc(m * n *sizeof(int), sizeof(int));
  for (i = 1; i < m-1; i++)
  {
    for (j = 1; j < n - 1; j++)
    {
      even[i*m+j] = rand() % 2;
    }
  }
  int *odd = (int *) calloc(m * n *sizeof(int), sizeof(int));
  write_output(even, m, n, res);

  dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 Grid(MATRIX_SIZE/BLOCK_SIZE,MATRIX_SIZE/BLOCK_SIZE);

  cudaMalloc((void **) &dev_even, m*n*sizeof(int));
  cudaMalloc((void **) &dev_odd, m*n*sizeof(int));
  cudaEventRecord(cstart, 0);
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
  cudaEventRecord(cstop, 0);
  cudaEventSynchronize(cstop);
  cudaEventElapsedTime(&time, cstart, cstop);
  
  fclose(res);
  free(even);
  free(odd);
  cudaFree(dev_even);
  cudaFree(dev_odd);
}
