#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ctime>
using namespace std;

double F(double x) {
	return (sin(x)-cos(x)+tan(x)-fabs(x-10000)+fabs(-x+5000));
}

double simpson_notomp(double h, int n, double* x) {
	double out = 0;
	int i;
	for (i = 1; i < n; i += 2) {
		out += F(x[i - 1]) + 4*F(x[i]) + F(x[i + 1]);
		}
	out *= h;
	return out;
}

double simpson_omp(double h, int n, double* x) {
	double out = 0;
	int i;
	#pragma omp parallel for shared(out) private(i) dynamic()
	{
		for (i = 1; i < n; i += 2) {
			out += F(x[i - 1]) + 4 * F(x[i]) + F(x[i + 1]);
		}
	}
	out *= h;
	return out;
}

double simpson_handomp(double h, int n, double* x) {
	double out = 0;
	int i;
	omp_set_num_threads(4);
#pragma omp parallel shared(i,out) dynamic()
	{
		int numt = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int nin1 = n / numt;
		for (i = (tid * n) + 1; i < (nin1 * (tid + 1)); i += 2) {
			out += F(x[i - 1]) + 4 * F(x[i]) + F(x[i + 1]);
		}
	}
	out *= h;
	return out;
}

int main() {
	int i, n=20;
	double a, b, h, *x, dtime, epsilon;
	x = (double*)malloc(sizeof(double) * n);
	epsilon = 0.0001;
	a = 1;
	b = 2;
	h = (b - a) / (n*3);
	cout << h << endl;
	for (i = 0; i < n; i++) x[i] = a + i * h;
	n = 2 * n - 1;
	dtime = omp_get_wtime();
	cout << simpson_notomp(h, n, x) << endl;
	dtime = omp_get_wtime() - dtime;
	printf("%f\n", dtime);

	dtime = omp_get_wtime();
	cout << simpson_omp(h,n,x) << endl;
	dtime = omp_get_wtime() - dtime;
	printf("%f\n", dtime);

	dtime = omp_get_wtime();
	cout << simpson_handomp(h,n,x) << endl;
	dtime = omp_get_wtime() - dtime;
	printf("%f\n", dtime);

	return 0;
}