#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ITER 20
#define BILLION 1000000000

void write_output(int **mat, int m, int n, FILE *res)
{
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      fprintf(res, "%d ", mat[i][j]);
    }
    fprintf(res, "\n");
  }
  for (int i = 0; i < 3; i++)
  {
    fprintf(res, "\n");
  }
}

void update_cell(int **current, int **future, int x, int y)
{
  int aliveN = 0;
  for (int i = -1; i <= 1; i++)
  {
    for (int j = -1; j <= 1; j++)
    {
      aliveN += current[x + i][y + j];
    }
  }
  aliveN -= current[x][y];

  //if lonely it dies
  if (aliveN < 2 && current[x][y] == 1)
    future[x][y] = 0;
  //if overpopulated it dies
  else if (aliveN > 3 && current[x][y] == 1)
  {
    future[x][y] = 0;
  }
  // if repopulated it revives
  else if (aliveN == 3 && current[x][y] == 0)
    future[x][y] = 1;
  // else copy current to future
  else
  {
    future[x][y] = current[x][y];
  }
}

void update_matrix(int **current, int **future, int m, int n)
{
  for (int i = 1; i < m - 1; i++)
  {
    for (int j = 1; j < n - 1; j++)
    {
      update_cell(current, future, i, j);
    }
  }
}

int main()
{
  struct timespec start, end;
  int i, j;
  int m, n;

  FILE *res;
  res = fopen("output.txt", "w");

  m = n = 64;

  srand(0);

  int **even = malloc(m * sizeof(int *));
  for (i = 0; i < m; i++)
  {
    even[i] = calloc(n * sizeof(int), sizeof(int));
    for (j = 1; j < n - 1; j++)
    {
      if (i == 0 || i == m - 1)
      {
        even[i][j] = 0;
      }
      else
        even[i][j] = rand() % 2;
    }
  }
  int **odd = malloc(sizeof(int *) * m);
  write_output(even, m, n, res);
  for (int i = 0; i < m; i++)
  {
    odd[i] = calloc(sizeof(int) * n, sizeof(int));
  }

  clock_getres(CLOCK_MONOTONIC, &start);
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int iter = 0; iter < ITER; iter++)
  {
    if (iter % 2 == 0)
    {
      update_matrix(even, odd, m, n);
      write_output(odd, m, n, res);
    }
    if (iter % 2 == 1)
    {
      update_matrix(odd, even, m, n);
      write_output(even, m, n, res);
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  float avg_time = (BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / ITER;
  printf("%f \n", avg_time);
  fclose(res);
  free(even);
  free(odd);
}
