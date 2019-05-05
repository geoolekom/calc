#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "dodulad_ci/ci.hpp"

#define N 16
#define PI 3.14159265358979323846
#define accomod 0.8
#define size_x 200
#define size_y 10
#define size_z 10
#define timesteps 50000

// size_x > size_y, size_z, else change the index for the array help when initializing

inline double length(double a, double b, double c) {
  return sqrt(a * a + b * b + c * c);
}

inline double square(double a) {
  return a * a;
}



double f_volna(int i, int gamma, double* f_pr, double ksi, int key, int size) {
  double theta, min;
  double z;
  double fn, f_1;

  fn = 2.0 * f_pr[size - 1] - f_pr[size - 2];
  f_1 = 2.0 * f_pr[0] - f_pr[1];

  if (fn < 0.0) {
    fn = 0.0;
  }
  if (f_1 < 0.0) {
    f_1 = 0.0;
  }

  int k;
  if (ksi < 0) {
    k = i + 1;
  }
  if (ksi > 0) {
    k = i;
  }


  if (ksi > 0) {
    if ((k > 0) && (k < size - 1)) {
      if ((f_pr[k + 1] - f_pr[k]) != 0.0) {
        z = (f_pr[k] - f_pr[k - 1]) / (f_pr[k + 1] - f_pr[k]);
      }
      else {
        z = (f_pr[k] - f_pr[k - 1]) / (f_pr[k + 1] - f_pr[k] + 1e-15);
      }
    }
    if (k == size - 1) {
      if (fn != f_pr[k]) {
        z = (f_pr[k] - f_pr[k - 1]) / (fn - f_pr[k]);
      }
      else {
        z = (f_pr[k] - f_pr[k - 1]) / (fn - f_pr[k] + 1e-15);
      }
    }
  }



  if (ksi < 0) {

    if ((i < size - 2) && (i != -1)) {
      if (f_pr[i + 1] != f_pr[i]) {
        z = (f_pr[i + 2] - f_pr[i + 1]) / (f_pr[i + 1] - f_pr[i]);
      }
      else {
        z = (f_pr[i + 2] - f_pr[i + 1]) / (f_pr[i + 1] - f_pr[i] + 1e-15);
      }
    }
    if (i == size - 2) {
      if (f_pr[i + 1] != f_pr[i]) {
        z = (fn - f_pr[i + 1]) / (f_pr[i + 1] - f_pr[i]);
      }
      else {
        z = (fn - f_pr[i + 1]) / (f_pr[i + 1] - f_pr[i] + 1e-15);
      }
    }
    if (i == -1) {
      if (f_pr[0] != f_1) {
        z = (f_pr[1] - f_pr[0]) / (f_pr[0] - f_1);
      }
      else {
        z = (f_pr[1] - f_pr[0]) / (f_pr[0] - f_1 + 1e-15);
      }
    }
  }

  if (z <= 1.0) {
    min = z;
  }
  else {
    min = 1.0;
  }
  if (min >= 0.0) {
    theta = min;
  }
  else {
    theta = 0.0;
  }



  if (key == 0) {
    if ((ksi > 0) && (i != size - 1)) {
      return (f_pr[k] + 0.5 * (1 - gamma) * (f_pr[i + 1] - f_pr[i]) * theta);
    }
    if ((ksi < 0) && (i != size - 1) && (i != -1)) {
      return (f_pr[k] - 0.5 * (1 - gamma) * (f_pr[i + 1] - f_pr[i]) * theta);
    }
    if ((ksi < 0) && (i == size - 1)) {
      return (f_pr[k] - 0.5 * (1 - gamma) * (fn - f_pr[i]) * theta);
    }
    if ((ksi < 0) && (i == - 1)) {
      return (f_pr[k] - 0.5 * (1 - gamma) * (f_pr[0] - f_1) * theta);
    }
    if ((ksi > 0) && (i == size - 1)) {
      return (f_pr[k] + 0.5 * (1 - gamma) * (fn - f_pr[i]) * theta);
    }
  }
  if (key == 1) {
    if ((ksi > 0) && (i != size - 1)) {
      return (f_pr[k] + 0.5 * (f_pr[i + 1] - f_pr[i]) * theta);
    }
    if ((ksi < 0) && (i != size - 1) && (i != -1)) {
      return (f_pr[k] - 0.5 * (f_pr[i + 1] - f_pr[i]) * theta);
    }
    if ((ksi < 0) && (i == size - 1)) {
      return (f_pr[k] - 0.5 * (fn - f_pr[i]) * theta);
    }
    if ((ksi < 0) && (i == - 1)) {
      return (f_pr[k] - 0.5 * (f_pr[0] - f_1) * theta);
    }
    if ((ksi > 0) && (i == size - 1)) {
      return (f_pr[k] + 0.5 * (fn - f_pr[i]) * theta);
    }
  }
  printf("probem with f_volna\n");
  return 0.0;
}




double* diff_right(double* help, double ksi_per, double ksi_1, double ksi_2, int size, double sum, double sum1, double sum_zv, double gamma) {

    double* result = (double*) malloc(2 * sizeof(double));

         double fn12 = (sum / sum1) * exp(- 0.5 * (square(length(ksi_per, ksi_1, ksi_2))));
         double fn = 2.0 * (sum_zv / sum1) * exp(-0.5 * (square(length(ksi_per, ksi_1, ksi_2)))) - help[size - 1];

         if (fn <= 0.0) {
           fn = 0.0;
         }

         double fn_112;

         if (help[size - 1] != help[size - 2]) {
           fn_112 = (fn - help[size - 1]) / (help[size - 1] - help[size - 2]);
         }
         else {
           fn_112 = (fn - help[size - 1]) / (help[size - 1] - help[size - 2] + 1e-15);
         }

         double z = fn_112;
         double min, theta;

         if (z <= 1.0) {
           min = z;
         }
         else {
           min = 1.0;
         }
         if (min >= 0.0) {
           theta = min;
         }
         else {
           theta = 0.0;
         }
         fn_112 = help[size - 1] - 0.5 * (1 - gamma) * (help[size - 1] - help[size - 2]) * theta;

         result[0] = help[size - 1] + gamma * (fn12 - fn_112);
         result[1] = help[size - 2] + gamma * (fn_112 - f_volna(size - 3, gamma, help, ksi_per, 0, size));

     return result;
}



double* diff_left(double* help, double ksi_per, double ksi_1, double ksi_2, int size, double summ, double summ1, double summ_zv, double gamma) {

    double* result = (double*) malloc(2 * sizeof(double));

         double f_12 = (summ / summ1) * exp(-0.5 * (square(length(ksi_per, ksi_1, ksi_2))));
         double f_1 = 2.0 * (summ_zv / summ1) * exp(-0.5 * (square(length(ksi_per, ksi_1, ksi_2)))) - help[0];

         if (f_1 <= 0.0) {
           f_1 = 0.0;
         }

         double f012;

         if (help[1] != help[0]) {
           f012 = (help[0] - f_1) / (help[1] - help[0]);
         }
         else {
           f012 = (help[0] - f_1) / (help[1] - help[0] + 1e-15);
         }

         double z = f012;
         double min, theta;

         if (z <= 1.0) {
           min = z;
         }
         else {
           min = 1.0;
         }
         if (min >= 0.0) {
           theta = min;
         }
         else {
           theta = 0.0;
         }

         f012 = help[0] + 0.5 * (1 - gamma) * (help[1] - help[0]) * theta;

         result[0] = help[0] - gamma * (f012 - f_12);
         result[1] = help[1] - gamma * (f_volna(1, gamma, help, ksi_per, 0, size) - f012);

   return result;
}







int main() 
{

  srand(time(NULL));

  double h, tau, gamma;
  double max;
  double x;
  double n2 = 10.0;
  double n1 = 1.0;
  double maxvell;
  double control1 = 0.0;
  double controli = 0.0;

  double speed[3] = {0.0, 0.0, 0.0};
  double temper[size_x][size_y];
  int imax;
  double temp_max = 0.0;
  double temp = 0.0;

  char fname_heatxy[] = "heat_xy/end_00000.txt";
  char fname_heatyz[] = "heat_yz/end_00000.txt";
  char fname_concxy[] = "conc_xy/end_00000.txt";
  char fname_tempxy[] = "temp_xy/end_00000.txt";
  char fname_heat_tempxy[] = "heat_temp_xy/end_00000.txt";
  char fname_heat_tempyz[] = "heat_temp_yz/end_00000.txt";
  char fname_frontxy[] = "front_xy/end_00000.txt";


  FILE *out = fopen(fname_heatxy, "w");
  FILE *out2 = fopen(fname_concxy, "w");
  FILE *out3 = fopen(fname_tempxy, "w");
  FILE *htemp = fopen(fname_heat_tempxy, "w");

  FILE *out_1 = fopen(fname_heatyz, "w");
  FILE *htemp_1 = fopen(fname_heat_tempyz, "w");


  double ecut = 5.26;
  double delta_ksi[3];
  int n_ksi[3] = {N, N, N};
  int i, j, l, k1, k2, k3;
  int size_y1 = int(size_y / 2.0);
  int size_z1 = int(size_z / 2.0);

// 0 < ksi * tau / h <= 1 (0.5) - условие устойчивости

  h = 1.0;
  tau = 1.0 / ecut / 3.0 * h * 0.3;


  max = h * (size_x - 1);
  printf("Max = %lf\n", max);


// ввод необходимых данных для интеграла столкновений

  ci::HSPotential potential;
  ci::init(&potential, ci::NO_SYMM);
  ci::Particle particle;
  particle.d = 1.;

  double radius = double(n_ksi[0]) / 2.0;
  double a = ecut / radius;

// массив, преобразовывающий три индекса скоростей в один - надо интегралу столкновений

  int*** xyz2i = (int***) malloc(n_ksi[0] * sizeof(int**));
  for (i = 0; i < n_ksi[0]; i ++) {
    xyz2i[i] = (int**) malloc (n_ksi[1] * sizeof(int*));
    for (j = 0; j < n_ksi[1]; j ++) {
      xyz2i[i][j] = (int*) malloc(n_ksi[2] * sizeof(int));
      for (k1 = 0; k1 < n_ksi[2]; k1 ++) {
        xyz2i[i][j][k1] = i + 2 * radius *(j + 2 * radius * k1);
      }
    }
  }

  double* f_integral = (double*) malloc(n_ksi[0] * n_ksi[1] * n_ksi[2] * sizeof(double));






// сетка в скоростном пространстве

  double** ks = (double**) malloc(3 * sizeof(double*));
  for (i = 0; i < 3; i ++) {
    ks[i] = (double*) malloc(sizeof(double) * n_ksi[i]);
    delta_ksi[i] = 2.0 * ecut / n_ksi[i];
//    printf("delta_ksi = %lf\n", delta_ksi[i]);
    for (j = 0; j < n_ksi[i]; j ++) {
      ks[i][j] = -ecut + (j + 0.5) * delta_ksi[i];
    }
  }




  double****** f = (double******) malloc(size_x * sizeof(double*****));
  double****** f_pr = (double******) malloc(size_x * sizeof(double*****));
  double****** bufer;
  double* help = (double*) malloc(size_x * sizeof(double));

  for (i = 0; i < size_x; i ++) {
    x = i * h;
    f[i] = (double*****) malloc(size_y * sizeof(double****));
    f_pr[i] = (double*****) malloc(size_y * sizeof(double****));

    for (j = 0; j < size_y; j ++) {
      f[i][j] = (double****) malloc(size_z * sizeof(double***));
      f_pr[i][j] = (double****) malloc(size_z * sizeof(double***));

      for (l = 0; l < size_z; l ++) {
        f[i][j][l] = (double***) malloc(N * sizeof(double**));
        f_pr[i][j][l] = (double***) malloc(N * sizeof(double**));


        maxvell = 0.0;

        for (k1 = 0; k1 < N; k1 ++) {
          f[i][j][l][k1] = (double**) malloc(N * sizeof(double*));
          f_pr[i][j][l][k1] = (double**) malloc(N * sizeof(double*));

          for (k2 = 0; k2 < N; k2 ++) {
            f[i][j][l][k1][k2] = (double*) malloc(N * sizeof(double));
            f_pr[i][j][l][k1][k2] = (double*) malloc(N * sizeof(double));

            for (k3 = 0; k3 < N; k3 ++) {
              if (x < (max / 10)) {
                f[i][j][l][k1][k2][k3] = n2 * exp(-0.5 * square(length(ks[0][k1], ks[1][k2], ks[2][k3])));
                maxvell = maxvell + exp(-0.5 * square(length(ks[0][k1], ks[1][k2], ks[2][k3])));
              }
              else {
                f[i][j][l][k1][k2][k3] = n1 * exp(-0.5 * square(length(ks[0][k1], ks[1][k2], ks[2][k3])));
                maxvell = maxvell + exp(-0.5 * square(length(ks[0][k1], ks[1][k2], ks[2][k3])));
              }
// нормировка идет на maxvell, тк нормирующая константа для функции распределения наxодится в теории интегрированием, а в сеточном виде
// - суммированием по всем значениям скорости
            }
          }
        }
        maxvell *= delta_ksi[0] * delta_ksi[1] * delta_ksi[2];
        for (k1 = 0; k1 < N; k1 ++) {
          for (k2 = 0; k2 < N; k2 ++) {
            for (k3 = 0; k3 < N; k3 ++) {
              f[i][j][l][k1][k2][k3] = f[i][j][l][k1][k2][k3] / maxvell;
              f_pr[i][j][l][k1][k2][k3] = f[i][j][l][k1][k2][k3];
            }
          }
        }
      }
    }
  }

  double cn;
  for (i = 0; i < size_x; i ++) {
    for (j = 0; j < size_y; j ++) {
      for (l = 0; l < size_z; l ++) {
        cn = 0.0;
        speed[0] = 0.0;
        speed[1] = 0.0;
        speed[2] = 0.0;
        temp = 0.0;

// вычисляем среднюю скорость молекул и микротемпературу, выводим температуру на срезе канала (2 Д графики)

        for (k1 = 0; k1 < N; k1 ++) {
          for (k2 = 0; k2 < N; k2 ++) {
            for (k3 = 0; k3 < N; k3 ++) {
              cn = cn + f_pr[i][j][l][k1][k2][k3];
              speed[0] = speed[0] + ks[0][k1] * f_pr[i][j][l][k1][k2][k3];
              speed[1] = speed[1] + ks[1][k2] * f_pr[i][j][l][k1][k2][k3];
              speed[2] = speed[2] + ks[2][k3] * f_pr[i][j][l][k1][k2][k3];
            }
          }
        }
        cn *= delta_ksi[0] * delta_ksi[1] * delta_ksi[2];
        speed[0] = speed[0] / cn * delta_ksi[0] * delta_ksi[1] * delta_ksi[2];
        speed[1] = speed[1] / cn * delta_ksi[0] * delta_ksi[1] * delta_ksi[2];
        speed[2] = speed[2] / cn * delta_ksi[0] * delta_ksi[1] * delta_ksi[2];
        control1 = control1 + cn;

        for (k1 = 0; k1 < N; k1 ++) {
          for (k2 = 0; k2 < N; k2 ++) {
            for (k3 = 0; k3 < N; k3 ++) {
              temp = temp + square(length(ks[0][k1] - speed[0], ks[1][k2] - speed[1], ks[2][k3] - speed[3])) * f_pr[i][j][l][k1][k2][k3];
            }
          }
        }
        temp = temp / (3.0 * cn) * delta_ksi[0] * delta_ksi[1] * delta_ksi[2];

        if (l == 5) {
          fprintf(out, "%lf %lf %lf\n", i * h, j * h, cn);
          fprintf(htemp, "%lf %lf %lf\n", i * h, j * h, temp);
// вывод для 2Д графиков, а не тепловых карт
          if (j == 5) {
            fprintf(out2, "%lf %lf\n", i * h, cn);
            fprintf(out3, "%lf %lf\n", i * h, temp);
          }
        }
// тепловые карты по yz с сечением х = 50
        if (i == 50) {
          fprintf(out_1, "%lf %lf %lf\n", j * h, l * h, cn);
          fprintf(htemp_1, "%lf %lf %lf\n", j * h, l * h, temp);
        }
      }
      fprintf(out_1, "\n");
      fprintf(htemp_1, "\n");
    }
    fprintf(out, "\n");
    fprintf(htemp, "\n");
  }

  fclose(out);
  fclose(out2);
  fclose(out3);
  fclose(htemp);
  fclose(out_1);
  fclose(htemp_1);




  double tau1 = tau / 2.0;
  double sum;
  double sum1;
  double sum_zv;
  double summ;
  double summ1;
  double summ_zv;

  double* returning;




// boundary conditions - mirror - for x and y -> f[0][j][k1][k2][k3] = f[-1][j][N - 1 - k1][k2][k3] for k1 > (N-1)/2, f[1][k1] = f[-2][N-1-k1]
// for negative velocities f[size_x-1][k1] = f[size_x][N-1-k1], f[size_x-2][k1] = f[size_x+1][N-1-k1] for k1 <= (N-1)/2
// we are going to calculate f[-1] and f[-2] and f[N], f{N+1], so we initialize the corresponded variables


  double***** f1x = (double*****) malloc(size_y * sizeof(double***));
  double***** f2x = (double*****) malloc(size_y * sizeof(double***));
  double***** fnx = (double*****) malloc(size_y * sizeof(double***));
  double***** fn1x = (double*****) malloc(size_y * sizeof(double***));
  
  for (j = 0; j < size_y; j ++) {
    f1x[j] = (double****) malloc(size_z * sizeof(double***));
    f2x[j] = (double****) malloc(size_z * sizeof(double***));
    fnx[j] = (double****) malloc(size_z * sizeof(double***));
    fn1x[j] = (double****) malloc(size_z * sizeof(double***));
    for (l = 0; l < size_z; l ++) {
      f1x[j][l] = (double***) malloc(N * sizeof(double**));
      f2x[j][l] = (double***) malloc(N * sizeof(double**));
      fnx[j][l] = (double***) malloc(N * sizeof(double**));
      fn1x[j][l] = (double***) malloc(N * sizeof(double**));
      for (k1 = 0; k1 < N; k1 ++) {
        f1x[j][l][k1] = (double**) malloc(N * sizeof(double*));
        f2x[j][l][k1] = (double**) malloc(N * sizeof(double*));
        fnx[j][l][k1] = (double**) malloc(N * sizeof(double*));
        fn1x[j][l][k1] = (double**) malloc(N * sizeof(double*));
        for (k2 = 0; k2 < N; k2 ++) {
          f1x[j][l][k1][k2] = (double*) malloc(N * sizeof(double));
          f2x[j][l][k1][k2] = (double*) malloc(N * sizeof(double));
          fnx[j][l][k1][k2] = (double*) malloc(N * sizeof(double));
          fn1x[j][l][k1][k2] = (double*) malloc(N * sizeof(double));
          for (k3 = 0; k3 < N; k3 ++) {
            f1x[j][l][k1][k2][k3] = f[0][j][l][k1][k2][k3];
            f2x[j][l][k1][k2][k3] = f[0][j][l][k1][k2][k3];
            fnx[j][l][k1][k2][k3] = f[size_x - 1][j][l][k1][k2][k3];
            fn1x[j][l][k1][k2][k3] = f[size_x - 1][j][l][k1][k2][k3];
          }
        }
      }
    }
  }

  double***** f1y = (double*****) malloc(size_x * sizeof(double****));
  double***** f2y = (double*****) malloc(size_x * sizeof(double****));

  for (i = 0; i < size_x; i ++) {
    f1y[i] = (double****) malloc(size_z * sizeof(double***));
    f2y[i] = (double****) malloc(size_z * sizeof(double***));
    for (l = 0; l < size_z; l ++) {
      f1y[i][l] = (double***) malloc(N * sizeof(double**));
      f2y[i][l] = (double***) malloc(N * sizeof(double**));
      for (k1 = 0; k1 < N; k1 ++) {
        f1y[i][l][k1] = (double**) malloc(N * sizeof(double*));
        f2y[i][l][k1] = (double**) malloc(N * sizeof(double*));
        for (k2 = 0; k2 < N; k2 ++) {
          f1y[i][l][k1][k2] = (double*) malloc(N * sizeof(double));
          f2y[i][l][k1][k2] = (double*) malloc(N * sizeof(double));
          for (k3 = 0; k3 < N; k3 ++) {
            f1y[i][l][k1][k2][k3] = f[i][0][l][k1][k2][k3];
            f2y[i][l][k1][k2][k3] = f[i][0][l][k1][k2][k3];
          }
        }
      }
    }
  }



  double***** f1z = (double*****) malloc(size_x * sizeof(double****));
  double***** f2z = (double*****) malloc(size_x * sizeof(double****));

  for (i = 0; i < size_x; i ++) {
    f1z[i] = (double****) malloc(size_y * sizeof(double***));
    f2z[i] = (double****) malloc(size_y * sizeof(double***));
    for (j = 0; j < size_y; j ++) {
      f1z[i][j] = (double***) malloc(N * sizeof(double**));
      f2z[i][j] = (double***) malloc(N * sizeof(double**));
      for (k1 = 0; k1 < N; k1 ++) {
        f1z[i][j][k1] = (double**) malloc(N * sizeof(double*));
        f2z[i][j][k1] = (double**) malloc(N * sizeof(double*));
        for (k2 = 0; k2 < N; k2 ++) {
          f1z[i][j][k1][k2] = (double*) malloc(N * sizeof(double));
          f2z[i][j][k1][k2] = (double*) malloc(N * sizeof(double));
          for (k3 = 0; k3 < N; k3 ++) {
            f1z[i][j][k1][k2][k3] = f[i][j][0][k1][k2][k3];
            f2z[i][j][k1][k2][k3] = f[i][j][0][k1][k2][k3];
          }
        }
      }
    }
  }


// boundary conditions - mirror - for x and y -> f[0][j][k1][k2][k3] = f[-1][j][N - 1 - k1][k2][k3] for k1 > (N-1)/2, f[1][k1] = f[-2][N-1-k1]
// for negative velocities f[size_x-1][k1] = f[size_x][N-1-k1], f[size_x-2][k1] = f[size_x+1][N-1-k1] for k1 <= (N-1)/2

  double v_12, v12, v1;
  double xa = 0.0;














  for (int jt = 1; jt < timesteps; jt ++) {

    controli = 0.0;

// шаг по х на тау/2

    for (j = 0; j < size_y; j ++) {
      for (l = 0; l < size_z; l ++) {
        for (k2 = 0; k2 < N; k2 ++) {
          for (k3 = 0; k3 < N; k3 ++) {

            sum = 0.0;
            sum1 = 0.0;
            sum_zv = 0.0;
            summ = 0.0;
            summ1 = 0.0;
            summ_zv = 0.0;

            for (k1 = 0; k1 < N; k1 ++) {
              if (ks[0][k1] > 0.0) {
                gamma = ks[0][k1] * tau1 / h;
              }
              else {
                gamma = -ks[0][k1] * tau1 / h;
              }

              for (int ii = 0; ii < size_x; ii ++) {
                help[ii] = f_pr[ii][j][l][k1][k2][k3];
              }
              if (ks[0][k1] > 0.0) {
                for (i = 2; i < size_x; i ++) {
                  f[i][j][l][k1][k2][k3] = help[i] - ks[0][k1] * tau1 * (f_volna(i, gamma, help, ks[0][k1], 0, size_x) - f_volna(i - 1, gamma, help, ks[0][k1], 0, size_x)) / h;
                }
// накопление сумм для диффузных гран условий

                sum = sum + ks[0][k1] * f_volna(size_x - 1, gamma, help, ks[0][k1], 0, size_x);
                sum_zv = sum_zv + ks[0][k1] * f_volna(size_x - 1, gamma, help, ks[0][k1], 1, size_x);
                summ1 = summ1 + ks[0][k1] * exp(-0.5 * (square(length(ks[0][k1], ks[1][k2], ks[2][k3]))));

// осуществление расчета зеркальных гран условий (левая граница, 0 и 1 индексы)

                if ((help[0] - f1x[j][l][k1][k2][k3]) * (f1x[j][l][k1][k2][k3] - f2x[j][l][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[0] - f1x[j][l][k1][k2][k3]);
                  if (xa > fabs(f1x[j][l][k1][k2][k3] - f2x[j][l][k1][k2][k3])) {
                    xa = fabs(f1x[j][l][k1][k2][k3] - f2x[j][l][k1][k2][k3]);
                  }
                  if ((help[0] - f1x[j][l][k1][k2][k3]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v_12 = f1x[j][l][k1][k2][k3] + 0.5 * (1.0 - gamma) * xa;

                if ((help[1] - help[0]) * (help[0] - f1x[j][l][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[1] - help[0]);
                  if (xa > fabs(help[0] - f1x[j][l][k1][k2][k3])) {
                    xa = fabs(help[0] - f1x[j][l][k1][k2][k3]);
                  }
                  if ((help[1] - help[0]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v12 = help[0] + 0.5 * (1.0 - gamma) * xa;

                f[0][j][l][k1][k2][k3] = help[0] - ks[0][k1] * tau1 / h * (v12 - v_12);
                f[1][j][l][k1][k2][k3] = help[1] - ks[0][k1] * tau1 / h * (f_volna(1, gamma, help, ks[0][k1], 0, size_x) - v12);

// сохранение значений фр для следующих итераций (зеркало, левая граница) ПОКА ОСТАВИТЬ, ПОТОМ ПРОВЕРИТЬ, НЕ СТОИТ ЛИ ИСПОЛЬЗОВАТЬ f ВМЕСТО f_pr после результата

                f1x[j][l][k1][k2][k3] = f_pr[0][j][l][N - 1 - k1][k2][k3];
                f2x[j][l][k1][k2][k3] = f_pr[1][j][l][N - 1 - k1][k2][k3];
              }



              if (ks[0][k1] < 0.0) {
                for (i = 0; i < size_x - 2; i ++) {
                  f[i][j][l][k1][k2][k3] = help[i] - ks[0][k1] * tau1  * (f_volna(i, gamma, help, ks[0][k1], 0, size_x) - f_volna(i - 1, gamma, help, ks[0][k1], 0, size_x)) / h;
                }

// накопление сумм для диффузных гран условий

                sum1 = sum1 - ks[0][k1] * exp(-0.5 * (square(length(ks[0][k1], ks[1][k2], ks[2][k3]))));
                summ = summ - ks[0][k1] * f_volna(-1, gamma, help, ks[0][k1], 0, size_x);
                summ_zv = summ_zv - ks[0][k1] * f_volna(-1, gamma, help, ks[0][k1], 1, size_x);

// расчет зеркальных гран условий (правая граница, size-1 и size-2 индексы)

                if ((fn1x[j][l][k1][k2][k3] - fnx[j][l][k1][k2][k3]) * (fnx[j][l][k1][k2][k3] - help[size_x - 1]) > 0.0) {
                  xa = fabs(fn1x[j][l][k1][k2][k3] - fnx[j][l][k1][k2][k3]);
                  if (xa > fabs(fnx[j][l][k1][k2][k3] - help[size_x - 1])) {
                    xa = fabs(fnx[j][l][k1][k2][k3] - help[size_x - 1]);
                  }
                  if ((fn1x[j][l][k1][k2][k3] - fnx[j][l][k1][k2][k3]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v1 = fnx[j][l][k1][k2][k3] - 0.5 * (1 - gamma) * xa;

                if ((fnx[j][l][k1][k2][k3] - help[size_x - 1]) * (help[size_x - 1] - help[size_x - 2]) > 0.0) {
                  xa = fabs(fnx[j][l][k1][k2][k3] - help[size_x - 1]);
                  if (xa > fabs(help[size_x - 1] - help[size_x - 2])) {
                    xa = fabs(help[size_x - 1] - help[size_x - 2]);
                  }
                  if ((fnx[j][l][k1][k2][k3] - help[size_x - 1]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v12 = help[size_x - 1] - 0.5 * (1 - gamma) * xa;

                f[size_x - 1][j][l][k1][k2][k3] = help[size_x - 1] - ks[0][k1] * tau1 / h * (v1 - v12);
                f[size_x - 2][j][l][k1][k2][k3] = help[size_x - 2] - ks[0][k1] * tau1 / h * (v12 - f_volna(size_x - 3, gamma, help, ks[0][k1], 0, size_x));

// сохранение значений для зеркальных гран условий для следующих итераций (правая граница)

                fnx[j][l][k1][k2][k3] = f_pr[size_x - 1][j][l][N - 1 - k1][k2][k3];
                fn1x[j][l][k1][k2][k3] = f_pr[size_x - 2][j][l][N - 1 - k1][k2][k3];
              }
            }


// добавление к зеркальным гранусловиям диффузных с учетом коэффициента аккомодации


            for (k1 = 0; k1 < N; k1 ++) {
              for (int ii = 0; ii < size_x; ii ++) {
                help[ii] = f_pr[ii][j][l][k1][k2][k3];
              }

              if (ks[0][k1] < 0.0) {
                gamma = -ks[0][k1] * tau1 / h;
                returning = diff_right(help, ks[0][k1], ks[1][k2], ks[2][k3], size_x, sum, sum1, sum_zv, gamma);
                f[size_x - 1][j][l][k1][k2][k3] = (1.0 - accomod) * f[size_x - 1][j][l][k1][k2][k3] + accomod * returning[0];
                f[size_x - 2][j][l][k1][k2][k3] = (1.0 - accomod) * f[size_x - 2][j][l][k1][k2][k3] + accomod * returning[1];
                free(returning);
              }

              if (ks[0][k1] > 0.0) {
                gamma = ks[0][k1] * tau1 / h;
                returning = diff_left(help, ks[0][k1], ks[1][k2], ks[2][k3], size_x, summ, summ1, summ_zv, gamma);
                f[0][j][l][k1][k2][k3] = (1.0 - accomod) * f[0][j][l][k1][k2][k3] + accomod * returning[0];
                f[1][j][l][k1][k2][k3] = (1.0 - accomod) * f[1][j][l][k1][k2][k3] + accomod * returning[1];
                free(returning);
              }
            }

          }
        }
      }
    }
    bufer = f_pr;
    f_pr = f;
    f = bufer;

//проверка того, что функция распределения всегда >= 0.0
/*
    for (i = 0; i < size_x; i ++) {
      for (j = 0; j < size_y; j ++) {
        for (l = 0; l < size_z; l ++) {
          for (k1 = 0; k1 < N; k1 ++) {
            for (k2 = 0; k2 < N; k2 ++) {
              for (k3 = 0; k3 < N; k3 ++) {
                  if (f_pr[i][j][l][k1][k2][k3] < 0.0) {
                    printf("problem1 0, time = %d, f[%d][%d][%d][%d][%d][%d] = %lf\n", jt, i, j, l, k1, k2, k3, f_pr[i][j][l][k1][k2][k3]);
                  }
              }
            }
          }
        }
      }
    }
*/








// шаг по оси у на тау/2


    for (i = 0; i < size_x; i ++) {
      for (l = 0; l < size_z; l ++) {

        for (k1 = 0; k1 < N; k1 ++) {
          for (k3 = 0; k3 < N; k3 ++) {

            summ = 0.0;
            summ1 = 0.0;
            summ_zv = 0.0;

            for (k2 = 0; k2 < N; k2 ++) {
              if (ks[1][k2] > 0.0) {
                gamma = ks[1][k2] * tau1 / h;
              }
              else {
                gamma = -ks[1][k2] * tau1 / h;
              }

              for (int ii = 0; ii < size_y; ii ++) {
                help[ii] = f_pr[i][ii][l][k1][k2][k3];
              }

              if (ks[1][k2] > 0.0) {
                for (j = 2; j < size_y1; j ++) {
                  f[i][j][l][k1][k2][k3] = help[j] - ks[1][k2] * tau1  * (f_volna(j, gamma, help, ks[1][k2], 0, size_y) - f_volna(j - 1, gamma, help, ks[1][k2], 0, size_y)) / h;
                }

 // накопление сумм для диффузных гранусловий

                summ1 = summ1 + ks[1][k2] * exp(-0.5 * (square(length(ks[0][k1], ks[1][k2], ks[2][k3]))));

// зеркальные гранусловия, которые потом соединятся с диффузными

                if ((help[0] - f1y[i][l][k1][k2][k3]) * (f1y[i][l][k1][k2][k3] - f2y[i][l][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[0] - f1y[i][l][k1][k2][k3]);
                  if (xa > fabs(f1y[i][l][k1][k2][k3] - f2y[i][l][k1][k2][k3])) {
                    xa = fabs(f1y[i][l][k1][k2][k3] - f2y[i][l][k1][k2][k3]);
                  }
                  if ((help[0] - f1y[i][l][k1][k2][k3]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v_12 = f1y[i][l][k1][k2][k3] + 0.5 * (1.0 - gamma) * xa;

                if ((help[1] - help[0]) * (help[0] - f1y[i][l][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[1] - help[0]);
                  if (xa > fabs(help[0] - f1y[i][l][k1][k2][k3])) {
                    xa = fabs(help[0] - f1y[i][l][k1][k2][k3]);
                  }
                  if ((help[1] - help[0]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v12 = help[0] + 0.5 * (1 - gamma) * xa;

                f[i][0][l][k1][k2][k3] = help[0] - ks[1][k2] * tau1 / h * (v12 - v_12);
                f[i][1][l][k1][k2][k3] = help[1] - ks[1][k2] * tau1 / h * (f_volna(1, gamma, help, ks[1][k2], 0, size_y) - v12);

                f1y[i][l][k1][k2][k3] = f_pr[i][0][l][k1][N - 1 - k2][k3];
                f2y[i][l][k1][k2][k3] = f_pr[i][1][l][k1][N - 1 - k2][k3];
              }


              if (ks[1][k2] < 0.0) {
                for (j = 0; j < size_y1; j ++) {
                  f[i][j][l][k1][k2][k3] = help[j] - ks[1][k2] * tau1  * (f_volna(j, gamma, help, ks[1][k2], 0, size_y) - f_volna(j - 1, gamma, help, ks[1][k2], 0, size_y)) / h;
                }
                summ = summ - ks[1][k2] * f_volna(-1, gamma, help, ks[1][k2], 0, size_y);
                summ_zv = summ_zv - ks[1][k2] * f_volna(-1, gamma, help, ks[1][k2], 1, size_y);
              }
            }

// соединение диффузных и зеркальных гранусловий с учетом коэффициента аккомодации

            for (k2 = 0; k2 < N; k2 ++) {
              for (int ii = 0; ii < size_y; ii ++) {
                help[ii] = f_pr[i][ii][l][k1][k2][k3];
              }

              if (ks[1][k2] > 0.0) {
                gamma = ks[1][k2] * tau1 / h;
                returning = diff_left(help, ks[1][k2], ks[0][k1], ks[2][k3], size_y, summ, summ1, summ_zv, gamma);

                f[i][0][l][k1][k2][k3] = (1.0 - accomod) * f[i][0][l][k1][k2][k3] + accomod * returning[0];
                f[i][1][l][k1][k2][k3] = (1.0 - accomod) * f[i][1][l][k1][k2][k3] + accomod * returning[1];

                free(returning);
              }
            }


            for (k2 = 0; k2 < N; k2 ++) {
              int alpha = 1;
              for (j = size_y1; j < size_y; j ++) {
                f[i][j][l][k1][k2][k3] = f[i][size_y1 - alpha][l][k1][N - 1 - k2][k3];
                alpha = alpha + 1;
              }
            }

          }
        }
      }
    }
    bufer = f_pr;
    f_pr = f;
    f = bufer;

//проверка того, что функция распределения всегда >= 0.0
/*
    for (i = 0; i < size_x; i ++) {
      for (j = 0; j < size_y; j ++) {
        for (l = 0; l < size_z; l ++) {
          for (k1 = 0; k1 < N; k1 ++) {
            for (k2 = 0; k2 < N; k2 ++) {
              for (k3 = 0; k3 < N; k3 ++) {
                if (f_pr[i][j][l][k1][k2][k3] < 0.0) {
                  printf("problem2 0, time = %d, f[%d][%d][%d][%d][%d][%d] = %lf\n", jt, i, j, l, k1, k2, k3, f_pr[i][j][l][k1][k2][k3]);
                }
              }
            }
          }
        }
      }
    }
*/






// перенос по z с tau/2

    for (i = 0; i < size_x; i ++) {
      for (j = 0; j < size_y; j ++) {

        for (k1 = 0; k1 < N; k1 ++) {
          for (k2 = 0; k2 < N; k2 ++) {

            summ = 0.0;
            summ1 = 0.0;
            summ_zv = 0.0;

            for (k3 = 0; k3 < N; k3 ++) {
              if (ks[2][k3] > 0.0) {
                gamma = ks[2][k3] * tau1 / h;
              }
              else {
                gamma = -ks[2][k3] * tau1 / h;
              }

              for (int ii = 0; ii < size_z; ii ++) {
                help[ii] = f_pr[i][j][ii][k1][k2][k3];
              }

              if (ks[2][k3] > 0.0) {
                for (l = 2; l < size_z1; l ++) {
                  f[i][j][l][k1][k2][k3] = help[l] - ks[2][k3] * tau1  * (f_volna(l, gamma, help, ks[2][k3], 0, size_z) - f_volna(l - 1, gamma, help, ks[2][k3], 0, size_z)) / h;
                }

 // накопление сумм для диффузных гранусловий

                summ1 = summ1 + ks[2][k3] * exp(-0.5 * (square(length(ks[0][k1], ks[1][k2], ks[2][k3]))));

// зеркальные гранусловия, которые потом соединятся с диффузными

                if ((help[0] - f1z[i][j][k1][k2][k3]) * (f1z[i][j][k1][k2][k3] - f2z[i][j][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[0] - f1z[i][j][k1][k2][k3]);
                  if (xa > fabs(f1z[i][j][k1][k2][k3] - f2z[i][j][k1][k2][k3])) {
                    xa = fabs(f1z[i][j][k1][k2][k3] - f2z[i][j][k1][k2][k3]);
                  }
                  if ((help[0] - f1z[i][j][k1][k2][k3]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v_12 = f1z[i][j][k1][k2][k3] + 0.5 * (1.0 - gamma) * xa;

                if ((help[1] - help[0]) * (help[0] - f1z[i][j][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[1] - help[0]);
                  if (xa > fabs(help[0] - f1z[i][j][k1][k2][k3])) {
                    xa = fabs(help[0] - f1z[i][j][k1][k2][k3]);
                  }
                  if ((help[1] - help[0]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v12 = help[0] + 0.5 * (1 - gamma) * xa;

                f[i][j][0][k1][k2][k3] = help[0] - ks[1][k2] * tau1 / h * (v12 - v_12);
                f[i][j][1][k1][k2][k3] = help[1] - ks[1][k2] * tau1 / h * (f_volna(1, gamma, help, ks[2][k3], 0, size_z) - v12);

                f1z[i][j][k1][k2][k3] = f_pr[i][j][0][k1][k2][N - 1 - k3];
                f2z[i][j][k1][k2][k3] = f_pr[i][j][1][k1][k2][N - 1 - k3];
              }


              if (ks[2][k3] < 0.0) {
                for (l = 0; l < size_z1; l ++) {
                  f[i][j][l][k1][k2][k3] = help[l] - ks[2][k3] * tau1  * (f_volna(l, gamma, help, ks[2][k3], 0, size_z) - f_volna(l - 1, gamma, help, ks[2][k3], 0, size_z)) / h;
                }
                summ = summ - ks[2][k3] * f_volna(-1, gamma, help, ks[2][k3], 0, size_z);
                summ_zv = summ_zv - ks[2][k3] * f_volna(-1, gamma, help, ks[2][k3], 1, size_z);
              }
            }

// соединение диффузных и зеркальных гранусловий с учетом коэффициента аккомодации

            for (k3 = 0; k3 < N; k3 ++) {
              for (int ii = 0; ii < size_z; ii ++) {
                help[ii] = f_pr[i][j][ii][k1][k2][k3];
              }

              if (ks[2][k3] > 0.0) {
                gamma = ks[2][k3] * tau1 / h;
                returning = diff_left(help, ks[2][k3], ks[0][k1], ks[1][k2], size_z, summ, summ1, summ_zv, gamma);

                f[i][j][0][k1][k2][k3] = (1.0 - accomod) * f[i][j][0][k1][k2][k3] + accomod * returning[0];
                f[i][j][1][k1][k2][k3] = (1.0 - accomod) * f[i][j][1][k1][k2][k3] + accomod * returning[1];

                free(returning);
              }
            }


            for (k3 = 0; k3 < N; k3 ++) {
              int alpha = 1;
              for (l = size_z1; l < size_z; l ++) {
                f[i][j][l][k1][k2][k3] = f[i][j][size_z1 - alpha][k1][k2][N - 1 - k3];
                alpha = alpha + 1;
              }
            }

          }
        }
      }
    }
    bufer = f_pr;
    f_pr = f;
    f = bufer;

//проверка того, что функция распределения всегда >= 0.0
/*
    for (i = 0; i < size_x; i ++) {
      for (j = 0; j < size_y; j ++) {
        for (l = 0; l < size_z; l ++) {
          for (k1 = 0; k1 < N; k1 ++) {
            for (k2 = 0; k2 < N; k2 ++) {
              for (k3 = 0; k3 < N; k3 ++) {
                if (f_pr[i][j][l][k1][k2][k3] < 0.0) {
                  printf("problem3_z 0, time = %d, f[%d][%d][%d][%d][%d][%d] = %lf\n", jt, i, j, l, k1, k2, k3, f_pr[i][j][l][k1][k2][k3]);
                }
              }
            }
          }
        }
      }
    }
*/







// столкновения с шагом тау


    ci::gen(tau, 50000, radius, radius, xyz2i, xyz2i, a, 1., 1., particle, particle);
    for (i = 0; i < size_x; i ++) {
      for (j = 0; j < size_y; j ++) {
        for (l = 0; l < size_z; l ++) {

          for (k1 = 0; k1 < N; k1 ++) {
            for (k2 = 0; k2 < N; k2 ++) {
              for (k3 = 0; k3 < N; k3 ++) {
                f_integral[xyz2i[k1][k2][k3]] = f_pr[i][j][l][k1][k2][k3];
              }
            }
          }

          ci::iter(f_integral, f_integral);

          for (k1 = 0; k1 < N; k1 ++) {
            for (k2 = 0; k2 < N; k2 ++) {
              for (k3 = 0; k3 < N; k3 ++) {
                f_pr[i][j][l][k1][k2][k3] = f_integral[xyz2i[k1][k2][k3]];
              }
            }
          }
        }
      }
    }

//проверка того, что функция распределения всегда >= 0.0
/*
    for (i = 0; i < size_x; i ++) {
      for (j = 0; j < size_y; j ++) {
        for (l = 0; l < size_z; l ++) {
          for (k1 = 0; k1 < N; k1 ++) {
            for (k2 = 0; k2 < N; k2 ++) {
              for (k3 = 0; k3 < N; k3 ++) {
                if (f_pr[i][j][l][k1][k2][k3] < 0.0) {
                  printf("problem3 0, time = %d, f[%d][%d][%d][%d][%d][%d] = %lf\n", jt, i, j, l, k1, k2, k3, f_pr[i][j][l][k1][k2][k3]);
                }
              }
            }
          }
        }
      }
    }
*/








// перенос по х с тау/2

    for (j = 0; j < size_y; j ++) {
      for (l = 0; l < size_z; l ++) {
        for (k2 = 0; k2 < N; k2 ++) {
          for (k3 = 0; k3 < N; k3 ++) {

            sum = 0.0;
            sum1 = 0.0;
            sum_zv = 0.0;
            summ = 0.0;
            summ1 = 0.0;
            summ_zv = 0.0;

            for (k1 = 0; k1 < N; k1 ++) {
              if (ks[0][k1] > 0.0) {
                gamma = ks[0][k1] * tau1 / h;
              }
              else {
                gamma = -ks[0][k1] * tau1 / h;
              }

              for (int ii = 0; ii < size_x; ii ++) {
                help[ii] = f_pr[ii][j][l][k1][k2][k3];
              }
              if (ks[0][k1] > 0.0) {
                for (i = 2; i < size_x; i ++) {
                  f[i][j][l][k1][k2][k3] = help[i] - ks[0][k1] * tau1 * (f_volna(i, gamma, help, ks[0][k1], 0, size_x) - f_volna(i - 1, gamma, help, ks[0][k1], 0, size_x)) / h;
                }
// накопление сумм для диффузных гран условий

                sum = sum + ks[0][k1] * f_volna(size_x - 1, gamma, help, ks[0][k1], 0, size_x);
                sum_zv = sum_zv + ks[0][k1] * f_volna(size_x - 1, gamma, help, ks[0][k1], 1, size_x);
                summ1 = summ1 + ks[0][k1] * exp(-0.5 * (square(length(ks[0][k1], ks[1][k2], ks[2][k3]))));

// осуществление расчета зеркальных гран условий (левая граница, 0 и 1 индексы)

                if ((help[0] - f1x[j][l][k1][k2][k3]) * (f1x[j][l][k1][k2][k3] - f2x[j][l][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[0] - f1x[j][l][k1][k2][k3]);
                  if (xa > fabs(f1x[j][l][k1][k2][k3] - f2x[j][l][k1][k2][k3])) {
                    xa = fabs(f1x[j][l][k1][k2][k3] - f2x[j][l][k1][k2][k3]);
                  }
                  if ((help[0] - f1x[j][l][k1][k2][k3]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v_12 = f1x[j][l][k1][k2][k3] + 0.5 * (1.0 - gamma) * xa;

                if ((help[1] - help[0]) * (help[0] - f1x[j][l][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[1] - help[0]);
                  if (xa > fabs(help[0] - f1x[j][l][k1][k2][k3])) {
                    xa = fabs(help[0] - f1x[j][l][k1][k2][k3]);
                  }
                  if ((help[1] - help[0]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v12 = help[0] + 0.5 * (1.0 - gamma) * xa;

                f[0][j][l][k1][k2][k3] = help[0] - ks[0][k1] * tau1 / h * (v12 - v_12);
                f[1][j][l][k1][k2][k3] = help[1] - ks[0][k1] * tau1 / h * (f_volna(1, gamma, help, ks[0][k1], 0, size_x) - v12);

// сохранение значений фр для следующих итераций (зеркало, левая граница) ПОКА ОСТАВИТЬ, ПОТОМ ПРОВЕРИТЬ, НЕ СТОИТ ЛИ ИСПОЛЬЗОВАТЬ f ВМЕСТО f_pr после результата

                f1x[j][l][k1][k2][k3] = f_pr[0][j][l][N - 1 - k1][k2][k3];
                f2x[j][l][k1][k2][k3] = f_pr[1][j][l][N - 1 - k1][k2][k3];
              }



              if (ks[0][k1] < 0.0) {
                for (i = 0; i < size_x - 2; i ++) {
                  f[i][j][l][k1][k2][k3] = help[i] - ks[0][k1] * tau1  * (f_volna(i, gamma, help, ks[0][k1], 0, size_x) - f_volna(i - 1, gamma, help, ks[0][k1], 0, size_x)) / h;
                }

// накопление сумм для диффузных гран условий

                sum1 = sum1 - ks[0][k1] * exp(-0.5 * (square(length(ks[0][k1], ks[1][k2], ks[2][k3]))));
                summ = summ - ks[0][k1] * f_volna(-1, gamma, help, ks[0][k1], 0, size_x);
                summ_zv = summ_zv - ks[0][k1] * f_volna(-1, gamma, help, ks[0][k1], 1, size_x);

// расчет зеркальных гран условий (правая граница, size-1 и size-2 индексы)

                if ((fn1x[j][l][k1][k2][k3] - fnx[j][l][k1][k2][k3]) * (fnx[j][l][k1][k2][k3] - help[size_x - 1]) > 0.0) {
                  xa = fabs(fn1x[j][l][k1][k2][k3] - fnx[j][l][k1][k2][k3]);
                  if (xa > fabs(fnx[j][l][k1][k2][k3] - help[size_x - 1])) {
                    xa = fabs(fnx[j][l][k1][k2][k3] - help[size_x - 1]);
                  }
                  if ((fn1x[j][l][k1][k2][k3] - fnx[j][l][k1][k2][k3]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v1 = fnx[j][l][k1][k2][k3] - 0.5 * (1 - gamma) * xa;

                if ((fnx[j][l][k1][k2][k3] - help[size_x - 1]) * (help[size_x - 1] - help[size_x - 2]) > 0.0) {
                  xa = fabs(fnx[j][l][k1][k2][k3] - help[size_x - 1]);
                  if (xa > fabs(help[size_x - 1] - help[size_x - 2])) {
                    xa = fabs(help[size_x - 1] - help[size_x - 2]);
                  }
                  if ((fnx[j][l][k1][k2][k3] - help[size_x - 1]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v12 = help[size_x - 1] - 0.5 * (1 - gamma) * xa;

                f[size_x - 1][j][l][k1][k2][k3] = help[size_x - 1] - ks[0][k1] * tau1 / h * (v1 - v12);
                f[size_x - 2][j][l][k1][k2][k3] = help[size_x - 2] - ks[0][k1] * tau1 / h * (v12 - f_volna(size_x - 3, gamma, help, ks[0][k1], 0, size_x));

// сохранение значений для зеркальных гран условий для следующих итераций (правая граница)

                fnx[j][l][k1][k2][k3] = f_pr[size_x - 1][j][l][N - 1 - k1][k2][k3];
                fn1x[j][l][k1][k2][k3] = f_pr[size_x - 2][j][l][N - 1 - k1][k2][k3];
              }
            }


// добавление к зеркальным гранусловиям диффузных с учетом коэффициента аккомодации


            for (k1 = 0; k1 < N; k1 ++) {
              for (int ii = 0; ii < size_x; ii ++) {
                help[ii] = f_pr[ii][j][l][k1][k2][k3];
              }

              if (ks[0][k1] < 0.0) {
                gamma = -ks[0][k1] * tau1 / h;
                returning = diff_right(help, ks[0][k1], ks[1][k2], ks[2][k3], size_x, sum, sum1, sum_zv, gamma);
                f[size_x - 1][j][l][k1][k2][k3] = (1.0 - accomod) * f[size_x - 1][j][l][k1][k2][k3] + accomod * returning[0];
                f[size_x - 2][j][l][k1][k2][k3] = (1.0 - accomod) * f[size_x - 2][j][l][k1][k2][k3] + accomod * returning[1];
                free(returning);
              }

              if (ks[0][k1] > 0.0) {
                gamma = ks[0][k1] * tau1 / h;
                returning = diff_left(help, ks[0][k1], ks[1][k2], ks[2][k3], size_x, summ, summ1, summ_zv, gamma);
                f[0][j][l][k1][k2][k3] = (1.0 - accomod) * f[0][j][l][k1][k2][k3] + accomod * returning[0];
                f[1][j][l][k1][k2][k3] = (1.0 - accomod) * f[1][j][l][k1][k2][k3] + accomod * returning[1];
                free(returning);
              }
            }

          }
        }
      }
    }
    bufer = f_pr;
    f_pr = f;
    f = bufer;

//проверка того, что функция распределения всегда >= 0.0
/*
    for (i = 0; i < size_x; i ++) {
      for (j = 0; j < size_y; j ++) {
        for (l = 0; l < size_z; l ++) {
          for (k1 = 0; k1 < N; k1 ++) {
            for (k2 = 0; k2 < N; k2 ++) {
              for (k3 = 0; k3 < N; k3 ++) {
                if (f_pr[i][j][l][k1][k2][k3] < 0.0) {
                  printf("problem4 0, time = %d, f[%d][%d][%d][%d][%d][%d] = %lf\n", jt, i, j, l, k1, k2, k3, f_pr[i][j][l][k1][k2][k3]);
                }
              }
            }
          }
        }
      }
    }
*/











// шаг по оси у на тау/2

    for (i = 0; i < size_x; i ++) {
      for (l = 0; l < size_z; l ++) {

        for (k1 = 0; k1 < N; k1 ++) {
          for (k3 = 0; k3 < N; k3 ++) {

            summ = 0.0;
            summ1 = 0.0;
            summ_zv = 0.0;

            for (k2 = 0; k2 < N; k2 ++) {
              if (ks[1][k2] > 0.0) {
                gamma = ks[1][k2] * tau1 / h;
              }
              else {
                gamma = -ks[1][k2] * tau1 / h;
              }

              for (int ii = 0; ii < size_y; ii ++) {
                help[ii] = f_pr[i][ii][l][k1][k2][k3];
              }

              if (ks[1][k2] > 0.0) {
                for (j = 2; j < size_y1; j ++) {
                  f[i][j][l][k1][k2][k3] = help[j] - ks[1][k2] * tau1  * (f_volna(j, gamma, help, ks[1][k2], 0, size_y) - f_volna(j - 1, gamma, help, ks[1][k2], 0, size_y)) / h;
                }

 // накопление сумм для диффузных гранусловий

                summ1 = summ1 + ks[1][k2] * exp(-0.5 * (square(length(ks[0][k1], ks[1][k2], ks[2][k3]))));

// зеркальные гранусловия, которые потом соединятся с диффузными

                if ((help[0] - f1y[i][l][k1][k2][k3]) * (f1y[i][l][k1][k2][k3] - f2y[i][l][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[0] - f1y[i][l][k1][k2][k3]);
                  if (xa > fabs(f1y[i][l][k1][k2][k3] - f2y[i][l][k1][k2][k3])) {
                    xa = fabs(f1y[i][l][k1][k2][k3] - f2y[i][l][k1][k2][k3]);
                  }
                  if ((help[0] - f1y[i][l][k1][k2][k3]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v_12 = f1y[i][l][k1][k2][k3] + 0.5 * (1.0 - gamma) * xa;

                if ((help[1] - help[0]) * (help[0] - f1y[i][l][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[1] - help[0]);
                  if (xa > fabs(help[0] - f1y[i][l][k1][k2][k3])) {
                    xa = fabs(help[0] - f1y[i][l][k1][k2][k3]);
                  }
                  if ((help[1] - help[0]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v12 = help[0] + 0.5 * (1 - gamma) * xa;

                f[i][0][l][k1][k2][k3] = help[0] - ks[1][k2] * tau1 / h * (v12 - v_12);
                f[i][1][l][k1][k2][k3] = help[1] - ks[1][k2] * tau1 / h * (f_volna(1, gamma, help, ks[1][k2], 0, size_y) - v12);

                f1y[i][l][k1][k2][k3] = f_pr[i][0][l][k1][N - 1 - k2][k3];
                f2y[i][l][k1][k2][k3] = f_pr[i][1][l][k1][N - 1 - k2][k3];
              }


              if (ks[1][k2] < 0.0) {
                for (j = 0; j < size_y1; j ++) {
                  f[i][j][l][k1][k2][k3] = help[j] - ks[1][k2] * tau1  * (f_volna(j, gamma, help, ks[1][k2], 0, size_y) - f_volna(j - 1, gamma, help, ks[1][k2], 0, size_y)) / h;
                }
                summ = summ - ks[1][k2] * f_volna(-1, gamma, help, ks[1][k2], 0, size_y);
                summ_zv = summ_zv - ks[1][k2] * f_volna(-1, gamma, help, ks[1][k2], 1, size_y);
              }
            }

// соединение диффузных и зеркальных гранусловий с учетом коэффициента аккомодации

            for (k2 = 0; k2 < N; k2 ++) {
              for (int ii = 0; ii < size_y; ii ++) {
                help[ii] = f_pr[i][ii][l][k1][k2][k3];
              }

              if (ks[1][k2] > 0.0) {
                gamma = ks[1][k2] * tau1 / h;
                returning = diff_left(help, ks[1][k2], ks[0][k1], ks[2][k3], size_y, summ, summ1, summ_zv, gamma);

                f[i][0][l][k1][k2][k3] = (1.0 - accomod) * f[i][0][l][k1][k2][k3] + accomod * returning[0];
                f[i][1][l][k1][k2][k3] = (1.0 - accomod) * f[i][1][l][k1][k2][k3] + accomod * returning[1];

                free(returning);
              }
            }


            for (k2 = 0; k2 < N; k2 ++) {
              int alpha = 1;
              for (j = size_y1; j < size_y; j ++) {
                f[i][j][l][k1][k2][k3] = f[i][size_y1 - alpha][l][k1][N - 1 - k2][k3];
                alpha = alpha + 1;
              }
            }

          }
        }
      }
    }
    bufer = f_pr;
    f_pr = f;
    f = bufer;

//проверка того, что функция распределения всегда >= 0.0
/*
    for (i = 0; i < size_x; i ++) {
      for (j = 0; j < size_y; j ++) {
        for (l = 0; l < size_z; l ++) {
          for (k1 = 0; k1 < N; k1 ++) {
            for (k2 = 0; k2 < N; k2 ++) {
              for (k3 = 0; k3 < N; k3 ++) {
                if (f_pr[i][j][l][k1][k2][k3] < 0.0) {
                  printf("problem5 0, time = %d, f[%d][%d][%d][%d][%d][%d] = %lf\n", jt, i, j, l, k1, k2, k3, f_pr[i][j][l][k1][k2][k3]);
                }
              }
            }
          }
        }
      }
    }
*/






// перенос по z с tau/2

    for (i = 0; i < size_x; i ++) {
      for (j = 0; j < size_y; j ++) {

        for (k1 = 0; k1 < N; k1 ++) {
          for (k2 = 0; k2 < N; k2 ++) {

            summ = 0.0;
            summ1 = 0.0;
            summ_zv = 0.0;

            for (k3 = 0; k3 < N; k3 ++) {
              if (ks[2][k3] > 0.0) {
                gamma = ks[2][k3] * tau1 / h;
              }
              else {
                gamma = -ks[2][k3] * tau1 / h;
              }

              for (int ii = 0; ii < size_z; ii ++) {
                help[ii] = f_pr[i][j][ii][k1][k2][k3];
              }

              if (ks[2][k3] > 0.0) {
                for (l = 2; l < size_z1; l ++) {
                  f[i][j][l][k1][k2][k3] = help[l] - ks[2][k3] * tau1  * (f_volna(l, gamma, help, ks[2][k3], 0, size_z) - f_volna(l - 1, gamma, help, ks[2][k3], 0, size_z)) / h;
                }

// накопление сумм для диффузных гранусловий

                summ1 = summ1 + ks[2][k3] * exp(-0.5 * (square(length(ks[0][k1], ks[1][k2], ks[2][k3]))));

// зеркальные гранусловия, которые потом соединятся с диффузными

                if ((help[0] - f1z[i][j][k1][k2][k3]) * (f1z[i][j][k1][k2][k3] - f2z[i][j][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[0] - f1z[i][j][k1][k2][k3]);
                  if (xa > fabs(f1z[i][j][k1][k2][k3] - f2z[i][j][k1][k2][k3])) {
                    xa = fabs(f1z[i][j][k1][k2][k3] - f2z[i][j][k1][k2][k3]);
                  }
                  if ((help[0] - f1z[i][j][k1][k2][k3]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v_12 = f1z[i][j][k1][k2][k3] + 0.5 * (1.0 - gamma) * xa;

                if ((help[1] - help[0]) * (help[0] - f1z[i][j][k1][k2][k3]) > 0.0) {
                  xa = fabs(help[1] - help[0]);
                  if (xa > fabs(help[0] - f1z[i][j][k1][k2][k3])) {
                    xa = fabs(help[0] - f1z[i][j][k1][k2][k3]);
                  }
                  if ((help[1] - help[0]) < 0.0) {
                    xa = -xa;
                  }
                } else {
                  xa = 0.0;
                }
                v12 = help[0] + 0.5 * (1 - gamma) * xa;

                f[i][j][0][k1][k2][k3] = help[0] - ks[1][k2] * tau1 / h * (v12 - v_12);
                f[i][j][1][k1][k2][k3] = help[1] - ks[1][k2] * tau1 / h * (f_volna(1, gamma, help, ks[2][k3], 0, size_z) - v12);

                f1z[i][j][k1][k2][k3] = f_pr[i][j][0][k1][k2][N - 1 - k3];
                f2z[i][j][k1][k2][k3] = f_pr[i][j][1][k1][k2][N - 1 - k3];
              }


              if (ks[2][k3] < 0.0) {
                for (l = 0; l < size_z1; l ++) {
                  f[i][j][l][k1][k2][k3] = help[l] - ks[2][k3] * tau1  * (f_volna(l, gamma, help, ks[2][k3], 0, size_z) - f_volna(l - 1, gamma, help, ks[2][k3], 0, size_z)) / h;
                }
                summ = summ - ks[2][k3] * f_volna(-1, gamma, help, ks[2][k3], 0, size_z);
                summ_zv = summ_zv - ks[2][k3] * f_volna(-1, gamma, help, ks[2][k3], 1, size_z);
              }
            }

// соединение диффузных и зеркальных гранусловий с учетом коэффициента аккомодации

            for (k3 = 0; k3 < N; k3 ++) {
              for (int ii = 0; ii < size_z; ii ++) {
                help[ii] = f_pr[i][j][ii][k1][k2][k3];
              }

              if (ks[2][k3] > 0.0) {
                gamma = ks[2][k3] * tau1 / h;
                returning = diff_left(help, ks[2][k3], ks[0][k1], ks[1][k2], size_z, summ, summ1, summ_zv, gamma);

                f[i][j][0][k1][k2][k3] = (1.0 - accomod) * f[i][j][0][k1][k2][k3] + accomod * returning[0];
                f[i][j][1][k1][k2][k3] = (1.0 - accomod) * f[i][j][1][k1][k2][k3] + accomod * returning[1];

                free(returning);
              }
            }


            for (k3 = 0; k3 < N; k3 ++) {
              int alpha = 1;
              for (l = size_z1; l < size_z; l ++) {
                f[i][j][l][k1][k2][k3] = f[i][j][size_z1 - alpha][k1][k2][N - 1 - k3];
                alpha = alpha + 1;
              }
            }

          }
        }
      }
    }
    bufer = f_pr;
    f_pr = f;
    f = bufer;

//проверка того, что функция распределения всегда >= 0.0
/*
    for (i = 0; i < size_x; i ++) {
      for (j = 0; j < size_y; j ++) {
        for (l = 0; l < size_z; l ++) {
          for (k1 = 0; k1 < N; k1 ++) {
            for (k2 = 0; k2 < N; k2 ++) {
              for (k3 = 0; k3 < N; k3 ++) {
                if (f_pr[i][j][l][k1][k2][k3] < 0.0) {
                  printf("problem3_z 0, time = %d, f[%d][%d][%d][%d][%d][%d] = %lf\n", jt, i, j, l, k1, k2, k3, f_pr[i][j][l][k1][k2][k3]);
                }
              }
            }
          }
        }
      }
    }
*/










    controli = 0.0;
    for (i = 0; i < size_x; i ++) {
      for (j = 0; j < size_y; j ++) {
        for (l = 0; l < size_z; l ++) {
          cn = 0.0;
          for (k1 = 0; k1 < N; k1 ++) {
            for (k2 = 0; k2 < N; k2 ++) {
              for (k3 = 0; k3 < N; k3 ++) {
                cn = cn + f_pr[i][j][l][k1][k2][k3];
              }
            }
          }
          cn *= delta_ksi[0] * delta_ksi[1] * delta_ksi[2];
          controli = controli + cn;
        }
      }
    }


    if (jt % 100 == 0) {
      sprintf(fname_heatxy, "heat_xy/end_%05u.txt", jt);
      sprintf(fname_concxy, "conc_xy/end_%05u.txt", jt);
      sprintf(fname_tempxy, "temp_xy/end_%05u.txt", jt);
      sprintf(fname_heat_tempxy, "heat_temp_xy/end_%05u.txt", jt);
      sprintf(fname_frontxy, "front_xy/end_%05u.txt", jt);

      sprintf(fname_heatyz, "heat_yz/end_%05u.txt", jt);
      sprintf(fname_heat_tempyz, "heat_temp_yz/end_%05u.txt", jt);
      FILE *out1 = fopen(fname_heatxy, "w");
      FILE *out2 = fopen(fname_concxy, "w");
      FILE *out3 = fopen(fname_tempxy, "w");
      FILE *htemp = fopen(fname_heat_tempxy, "w");
      FILE *front = fopen(fname_frontxy, "w");

      FILE *out11 = fopen(fname_heatyz, "w");
      FILE *htemp1 = fopen(fname_heat_tempyz, "w");

      for (i = 0; i < size_x; i ++) {
        for (j = 0; j < size_y; j ++) {
          for (l = 0; l < size_z; l ++) {
            cn = 0.0;
            speed[0] = 0.0;
            speed[1] = 0.0;
            speed[2] = 0.0;
            temp = 0.0;

            for (k1 = 0; k1 < N; k1 ++) {
              for (k2 = 0; k2 < N; k2 ++) {
                for (k3 = 0; k3 < N; k3 ++) {
                  cn = cn + f_pr[i][j][l][k1][k2][k3];
                  speed[0] = speed[0] + ks[0][k1] * f_pr[i][j][l][k1][k2][k3];
                  speed[1] = speed[1] + ks[1][k2] * f_pr[i][j][l][k1][k2][k3];
                  speed[2] = speed[2] + ks[2][k3] * f_pr[i][j][l][k1][k2][k3];
                }
              }
            }
            cn *= delta_ksi[0] * delta_ksi[1] * delta_ksi[2];
            speed[0] = speed[0] / cn * delta_ksi[0] * delta_ksi[1] * delta_ksi[2];
            speed[1] = speed[1] / cn * delta_ksi[0] * delta_ksi[1] * delta_ksi[2];
            speed[2] = speed[2] / cn * delta_ksi[0] * delta_ksi[1] * delta_ksi[2];

            for (k1 = 0; k1 < N; k1 ++) {
              for (k2 = 0; k2 < N; k2 ++) {
                for (k3 = 0; k3 < N; k3 ++) {
                  temp = temp + square(length(ks[0][k1] - speed[0], ks[1][k2] - speed[1], ks[2][k3] - speed[3])) * f_pr[i][j][l][k1][k2][k3];
                }
              }
            }
            temp = temp / (3.0 * cn) * delta_ksi[0] * delta_ksi[1] * delta_ksi[2];

            if (l == 5) {
              temper[i][j] = temp;
            }



            if ((j == 5) and (l == 5)) {
              fprintf(out2, "%lf %lf\n", i * h, cn);
              fprintf(out3, "%lf %lf\n", i * h, temp);
            }

            if (l == 5) {
            fprintf(out1, "%lf %lf %lf\n", i * h, j * h, cn);
            fprintf(htemp, "%lf %lf %lf\n", i * h, j * h, temp);
            }

            if (i == 50) {
              fprintf(out11, "%lf %lf %lf\n", i * h, j * h, cn);
              fprintf(htemp1, "%lf %lf %lf\n", i * h, j * h, temp);
            }
          }
          fprintf(out11, "\n");
          fprintf(htemp1, "\n");
        }
        fprintf(out1, "\n");
        fprintf(htemp, "\n");
      }


// поиск фронта УВ в плоскости z = 5 по максимуму продольной температуры

      for (j = 0; j < size_y; j ++) {
          temp_max = 0.0;
          imax = 0;
          for (i = 0; i < size_x; i ++) {
            if (temper[i][j] >= temp_max) {
              temp_max = temper[i][j];
              imax = i;
             }
          }
          fprintf(front, "%lf %lf\n", imax * h, j * h);
      }


      fclose(out1);
      fclose(out2);
      fclose(out3);
      fclose(htemp);
      fclose(front);
      fclose(out11);
      fclose(htemp1);
    }



    printf("%d\n", jt);

    if (fabs(control1 - controli)/control1 > 1e-6) {
      printf("problem with total concentration\n");
      if (control1 > controli) {
          printf("          downward trend: %lf %lf %lf\n", control1, controli, control1 - controli);
      }
      else {
          printf("          upward trend: %lf %lf %lf\n", control1, controli, controli - control1);
      }
    }


// end of step of time
  }






//delete all


  for (i = 0; i < size_x; i ++) {
    for (j = 0; j < size_y; j ++) {
      for (l = 0; l < size_z; l ++) {
        for (k1 = 0; k1 < N; k1 ++) {
          for (k2 = 0; k2 < N; k2 ++) {
            free(f[i][j][l][k1][k2]);
            free(f_pr[i][j][l][k1][k2]);
          }
        }
      }
    }
  }
  for (i = 0; i < size_x; i ++) {
    for (j = 0; j < size_y; j ++) {
      for (l = 0; l < size_z; l ++) {
        for (k1 = 0; k1 < N; k1 ++) {
          free(f[i][j][l][k1]);
          free(f_pr[i][j][l][k1]);
        }
      }
    }
  }
  for (i = 0; i < size_x; i ++) {
    for (j = 0; j < size_y; j ++) {
      for (l = 0; l < size_z; l ++) {
        free(f[i][j][l]);
        free(f_pr[i][j][l]);
      }
    }
  }
  for (i = 0; i < size_x; i ++) {
    for (j = 0; j < size_y; j ++) {
      free(f[i][j]);
      free(f_pr[i][j]);
    }
  }
  for (i = 0; i < size_x; i ++) {
    free(f[i]);
    free(f_pr[i]);
    if (i < 3) {
      free(ks[i]);
    }
  }
  free(help);
  free(f);
  free(f_pr);
  free(ks);

  free(f_integral);

  for (j = 0; j < size_y; j ++) {
    for (l = 0; l < size_z; l ++) {
      for (k1 = 0; k1 < N; k1 ++) {
        for (k2 = 0; k2 < N; k2 ++) {
          free(f1x[j][l][k1][k2]);
          free(f2x[j][l][k1][k2]);
          free(fnx[j][l][k1][k2]);
          free(fn1x[j][l][k1][k2]);
        }
      }
    }
  }
  for (j = 0; j < size_y; j ++) {
    for (l = 0; l < size_z; l ++) {
      for (k1 = 0; k1 < N; k1 ++) {
        free(f1x[j][l][k1]);
        free(f2x[j][l][k1]);
        free(fnx[j][l][k1]);
        free(fn1x[j][l][k1]);
      }
    }
  }
  for (j = 0; j < size_y; j ++) {
    for (l = 0; l < size_z; l ++) {
      free(f1x[j][l]);
      free(f2x[j][l]);
      free(fnx[j][l]);
      free(fn1x[j][l]);
    }
  }
  for (j = 0; j < size_y; j ++) {
    free(f1x[j]);
    free(f2x[j]);
    free(fnx[j]);
    free(fn1x[j]);
  }
  free(f1x);
  free(f2x);
  free(fnx);
  free(fn1x);

  for (j = 0; j < size_x; j ++) {
    for (l = 0; l < size_z; l ++) {
      for (k1 = 0; k1 < N; k1 ++) {
        for (k2 = 0; k2 < N; k2 ++) {
          free(f1y[j][l][k1][k2]);
          free(f2y[j][l][k1][k2]);
        }
      }
    }
  }
  for (j = 0; j < size_x; j ++) {
    for (l  = 0; l < size_z; l ++) {
      for (k1 = 0; k1 < N; k1 ++) {
        free(f1y[j][l][k1]);
        free(f2y[j][l][k1]);
      }
    }
  }
  for (j = 0; j < size_x; j ++) {
    for (l = 0; l < size_z; l ++) {
      free(f1y[j][l]);
      free(f2y[j][l]);
    }
  }
  for (j = 0; j < size_x; j ++) {
    free(f1y[j]);
    free(f2y[j]);
  }
  free(f1y);
  free(f2y);    


  for (i = 0; i < size_x; i ++) {
    for (j = 0; j < size_y; j ++) {
      for (k1 = 0; k1 < N; k1 ++) {
        for (k2 = 0; k2 < N; k2 ++) {
          free(f1z[i][j][k1][k2]);
          free(f2z[i][j][k1][k2]);
        }
      }
    }
  }
  for (i = 0; i < size_x; i ++) {
    for (j = 0; j < size_y; j ++) {
      for (k1 = 0; k1 < N; k1 ++) {
        free(f1z[i][j][k1]);
        free(f2z[i][j][k1]);
      }
    }
  }
  for (i = 0; i < size_x; i ++) {
    for (j = 0; j < size_y; j ++) {
      free(f1z[i][j]);
      free(f2z[i][j]);
    }
  }
  for (i = 0; i < size_x; i ++) {
    free(f1z[i]);
    free(f2z[i]);
  }
  free(f1z);
  free(f2z);


  for (i = 0; i < n_ksi[0]; i ++) {
    for (j = 0; j < n_ksi[1]; j ++) {
      free(xyz2i[i][j]);
    }
  }
  for (i = 0; i < n_ksi[0]; i ++) {
    free(xyz2i[i]);
  }
  free(xyz2i);

  return 0;
}
