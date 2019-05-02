#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ctime>
#include <math.h>
using namespace std;

double* yacobi_notomp(double* A, double* B, double* X, int n, double epsilon) {
	double* TempX, norm = 0, c;
	int i, j;
		TempX = (double*)malloc(sizeof(double) * n);
		do {
			for (i = 0; i < n; i++) {
				TempX[i] = B[i];
				for (j = 0; j < n; j++) {
					if (i != j)
						TempX[i] -= A[i * n + j] * X[j];
				}
				TempX[i] /= A[i * n + i];
				c = fabs(X[i] - TempX[i]);
				if (c < norm)
					norm = c;
			}
			for (i = 0; i < n; i++) {
				X[i] = TempX[i];
			}
		} while (norm > epsilon);
		return X;
}

double* yacobi_omp(double *A, double* B, double* X, int n, double epsilon) {
	double* TempX, norm = 0, c;
	int i, j;
		TempX = (double*)malloc(sizeof(double) * n);
		do {
#pragma omp parallel for shared(i,A,B,X,n,TempX,norm,c) / private(j) / dynamic()
			{
				for (i = 0; i < n; i++) {
					TempX[i] = B[i];
					for (j = 0; j < n; j++) {
						if (i != j)
							TempX[i] -= A[i * n + j] * X[j];
					}
					TempX[i] /= A[i * n + i];
					c = fabs(X[i] - TempX[i]);
					if (c < norm)
						norm = c;
				}
			}
#pragma omp parallel for shared(i,X,n,TempX) / dynamic()
			{
				for (i = 0; i < n; i++) {
					X[i] = TempX[i];
				}
			}
		} while (norm > epsilon);
		return X;
}

double* yacobi_handomp(double *A, double *B, double* X, int n, double epsilon) {
	double* TempX, norm = 0, c;
	int i, j;
		TempX = (double*)malloc(sizeof(double) * n);
		do {
#pragma omp parallel num_threads(4) / shared(i,A,B,X,n,TempX,norm,c) / private(j)
			{
				int numt = omp_get_num_threads();
				int tid = omp_get_thread_num();
				int nim = n / numt;
				for (i = (tid * n); i < (nim * (tid + 1)); i++) {
					TempX[i] = B[i];
					for (j = 0; j < n; j++) {
						if (i != j)
							TempX[i] -= A[i * n + j] * X[j];
					}
					TempX[i] /= A[i * n + i];
					c = fabs(X[i] - TempX[i]);
					if (c < norm)
						norm = c;
				}
			}
#pragma omp parallel num_threads(4) / shared(i,X,n,TempX)
			{
				int numt = omp_get_num_threads();
				int tid = omp_get_thread_num();
				int nim = n / numt;
				for (i = (tid * n); i < (nim * (tid + 1)); i++) {
					X[i] = TempX[i];
				}
			}
		} while (norm > epsilon);
		return X;
}



int main() {
	int i, n;
	double* A, * B, * X, dtime, epsilon, norm = 0;
	epsilon = 0.001;
	n = 10000;
	A = (double*)malloc(sizeof(double) * n * n);
	B = (double*)malloc(sizeof(double) * n);
	X = (double*)malloc(sizeof(double) * n);
	for (i = 0; i < n * n; i++) { A[i] = rand(); }
	do {
		for (i = 0; i < n; i++) { B[i] = (cos(rand()) - sin(rand()))/n; }
		for (i = 0; i < n; i++) { norm += B[i] * B[i]; };
		norm = sqrt(norm);
	} while (norm >= 1);
	X = B;
	dtime = omp_get_wtime();
	yacobi_notomp(A, B, X, n, epsilon);
	dtime = omp_get_wtime() - dtime;
	printf("%f\n", dtime);
	X = B;
	//for (i = 0; i < n; i++) { cout << C[i] << endl; };

	dtime = omp_get_wtime();
	yacobi_omp(A, B, X, n, epsilon);
	dtime = omp_get_wtime() - dtime;
	printf("%f\n", dtime);
	X = B;
	//for (i = 0; i < n; i++) { cout << C[i] << endl; };

	dtime = omp_get_wtime();
	yacobi_handomp(A, B, X, n, epsilon);
	dtime = omp_get_wtime() - dtime;
	printf("%f\n", dtime);
	//for (i = 0; i < n; i++) { cout << C[i] << endl; };

}