#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <assert.h>

#define ITER 20
#define BILLION 1000000000
#define BLOCK_SIZE 8 // max is 32

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
  
  __shared__ int curr_shared[BLOCK_SIZE*BLOCK_SIZE];
  
  curr_shared[x*m+y] = current[x*m+y];

  __syncthreads();

  if (x >= m-1 || y >= n-1) future[x*m + y] = 0;
  else if (x==0 || y == 0) future[x*m + y] = 0;
  else{
    for (int i = 1; i < m - 1; i++)
    {
      for (int j = 1; j < n - 1; j++)
      {
      int aliveN = 0;
      for (int i = -1; i <= 1; i++)
      {
          for (int j = -1; j <= 1; j++)
          {
          aliveN += curr_shared[(x + i)*m + y + j];
          }
      }
      aliveN -= curr_shared[x*m + y];
      __syncthreads();
      //if lonely it dies
      if (aliveN < 2 && curr_shared[x*m + y] == 1) {
        future[x*m + y] = 0;
      }
      //if overpopulated it dies
      else if (aliveN > 3 && curr_shared[x*m + y] == 1)
      {
        future[x*m + y] = 0;
      }
      // if repopulated it revives
      else if (aliveN == 3 && curr_shared[x*m + y] == 0) {
        future[x*m + y] = 1;
      }
      // else copy current to future
      else
      {
        future[x*m + y] = curr_shared[x*m + y];
      }
      }
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
  res = fopen("output_CUDA.txt", "w");

  m = n = 8;

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

  dim3 Block(m,n);
  dim3 Grid(1,1);

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
}
