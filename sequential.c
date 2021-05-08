#include <stdio.h>
#include <stdlib.h>

int main()
{
  int i, j;
  int m, n;

  FILE *file;
  file = fopen("MyMatrix.txt", "r");

  m = n = 8;

  double **mat = malloc(m * sizeof(double *));
  for (i = 0; i < m; ++i)
    mat[i] = malloc(n * sizeof(double));

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      if (!fscanf(file, "%lf", &mat[i][j]))
        break;
  printf("%.16lf, %.16lf\n", mat[0][0], mat[8][8]);
  fclose(file);

  int *
  // Loop through every cell
}
