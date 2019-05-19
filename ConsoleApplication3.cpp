#include "pch.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ctime>
using namespace std;

/*
Matrix sizes input with validation check
Returns four matrix sizes from 1 to 65,535 aka unsigned short int without 0
*/
void size_validation_input(unsigned short int& n1, unsigned short int& n2, unsigned short int& n3, unsigned short int& n4) { //Matrix sizes check
	bool input_error;
	cout << "Input number of rows in first matrix:" << endl;
	cin >> n1;
	while (cin.fail() || n1 == 0) {
		cout << "Error! Enter a number beetwen 0 and 65,536:" << endl;
		cin.clear();
		cin.ignore(numeric_limits<streamsize>::max(), '\n');
		cin >> n1;
	}
	do {
		cout << "Input number of columns in first matrix:" << endl;
		cin >> n2;
		while (cin.fail() || n2 == 0) {
			cout << "Error! Enter a number beetwen 0 and 65,536:" << endl;
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			cin >> n2;
		}
		cout << "Input number of rows in second matrix:" << endl;
		cin >> n3;
		while (cin.fail() || n3 == 0) {
			cout << "Error! Enter a number beetwen 0 and 65,536:" << endl;
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			cin >> n3;
		}
		if (n2 != n3) {
			cout << "Critical error! Number of columns in first matrix is not equal to number of rows in second matrix. Matrix multiplication impossible. Please rewrite your inputs." << endl;
			input_error = true;
		}
		else input_error = false;
	} while (input_error);
	cout << "Input number of columns in second matrix:" << endl;
	cin >> n4;
	while (cin.fail() || n4 == 0) {
		cout << "Error! Enter a number beetwen 0 and 65,536:" << endl;
		cin.clear();
		cin.ignore(numeric_limits<streamsize>::max(), '\n');
		cin >> n4;
	}
}

/*
Boolean input string validation
Checks if user use 0, 1, true, false, TrUE, FalSe, y, yes, no, n, etc as input and converts it into boolean value.
*/
bool boolean_validation_input() {
	unsigned short int i = 0;
	bool error, result;
	char input_str [5];
	do {
		cin >> input_str;
		error = false;
		while ((cin.fail() || strlen(input_str) > 5) && (input_str != NULL) && (input_str[0] != '\0')) {
			cout << "Invalid input! Use valid inputs instead: 1, 0, y, n, yes, no, true, false." << endl;
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			cin >> input_str;
		}
		for (i = 0; i < 5; i++) {
			input_str[i] = tolower(input_str[i]);
		}
		if ((strcmp(input_str, "1") == 0) || (strcmp(input_str, "yes") == 0) || (strcmp(input_str, "y") == 0) || (strcmp(input_str, "true") == 0)) {
			result = true;
		}
		else {
			if ((strcmp(input_str, "0") == 0) || (strcmp(input_str, "no") == 0) || (strcmp(input_str, "n") == 0) || (strcmp(input_str, "false") == 0)) {
				result = false;
			}
			else {
				error = true;
			}
		}
		if (error) {
			cout << "Invalid input! Use valid inputs instead: 1, 0, y, n, yes, no, true, false." << endl;
		}
	} while (error);
	return result;
}

/*
Double number input validation
*/
double double_input_validation() {
	double input;
	cin >> input;
	while (cin.fail()) {
		cout << "Error! Enter a valid double number:" << endl;
		cin.clear();
		cin.ignore(numeric_limits<streamsize>::max(), '\n');
		cin >> input;
	}
	return input;
}

/*
Natural number input validation
*/
unsigned int natural_input_validation() {
	unsigned int input;
	cin >> input;
	while (cin.fail() || input == 0) {
		cout << "Error! Enter a valid natural number:" << endl;
		cin.clear();
		cin.ignore(numeric_limits<streamsize>::max(), '\n');
		cin >> input;
	}
	return input;
}

/*
Matrix multiplication function
A - first matrix, B - second matrix, C - output matrix
n1 - rows in A, n23 - columns/rows in A/B, n4 - columns in B
*/
void matrix_mult(double* A, double* B, double*& C, int n1, int n23, int n4)
{
	int i, j, k;
	double dot;
	for (i = 0; i < n1; i++) {
		for (j = 0; j < n4; j++) {
			dot = 0;
			for (k = 0; k < n23; k++) {
				dot += A[i*n23 + k] * B[k*n4 + j];
			}
			C[i*n4 + j] = dot;
		}
	}
}

/*
Matrix multiplication function with multithreading
A - first matrix, B - second matrix, C - output matrix
n1 - rows in A, n23 - columns/rows in A/B, n4 - columns in B
*/
void matrix_mult_omp(double* A, double* B, double*& C, int n1, int n23, int n4)
{
		int i, j, k;
#pragma omp parallel for shared(A,B,C,n1,n23,n4,i) private(j,k) schedule(dynamic,1)
				for (i = 0; i < n1; i++) {
					for (j = 0; j < n4; j++) {
						double dot = 0;
						for (k = 0; k < n23; k++) {
							dot += A[i*n23 + k] * B[k*n4 + j];
						}
						C[i*n4 + j] = dot;
					}
				}
}

/*
Matrix multiplication function with block realisathion of multithreading
A - first matrix, B - second matrix, C - output matrix
n1 - rows in A, n23 - columns/rows in A/B, n4 - columns in B
*/
void matrix_mult_block_omp(double* A, double* B, double*& C, int n1, int n23, int n4)
{
	int i, j, k, it;
#pragma omp parallel num_threads(5) shared(A,B,C,n1,n23,n4) private(j,k,i,it)
	{
		int numt = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int number = sqrt(numt);
		int nin1 = n1 / number;
		int rowIndex = tid / nin1;
		int colIndex = tid % nin1;
		for (it = 0; it < number; it++)
		{
			for ( i = rowIndex * nin1; i < (rowIndex + 1) * nin1; i++)
				for ( j = colIndex * nin1; j < (colIndex + 1) * nin1; j++)
				{
					double dot = 0;
					for (k = it * nin1; k < (it + 1) * nin1; k++) {
						dot += A[i * n1 + k] * B[k * n4 + j];
					}
					C[i * n23 + j] = dot;
				}
		}
	}
}

int main() {
	unsigned short int n1, n2, n3, n4;
	unsigned int i, foraverage = 1;
	double *A, *B, *C, dtime, averagetime;
	bool test = false;
	cout << "Test performance? No, for custom input." << endl;
	if (boolean_validation_input()) {
		n1 = 3; n2 = 3; n3 = 3; n4 = 3;
		A = (double*)malloc(sizeof(double) * n1 * n2);
		B = (double*)malloc(sizeof(double) * n3 * n4);
		C = (double*)malloc(sizeof(double) * n1 * n4);
		A[0] = 2; A[1] = 2; A[2] = 2; A[3] = 5; A[4] = 4; A[5] = 3; A[6] = 3; A[7] = 8; A[8] = 4;
		B[0] = 2; B[1] = 2; B[2] = 2; B[3] = 5; B[4] = 4; B[5] = 3; B[6] = 3; B[7] = 8; B[8] = 4;
		test = true;
	}
	else {
		cout << "Enter matrix sizes:" << endl;
		size_validation_input(n1, n2, n3, n4);
		A = (double*)malloc(sizeof(double) * n1 * n2);
		B = (double*)malloc(sizeof(double) * n3 * n4);
		C = (double*)malloc(sizeof(double) * n1 * n4);
		cout << "Fill with random numbers?" << endl;
		if (boolean_validation_input()) {
			for (i = 0; i < n1 * n2; i++) A[i] = rand();
			for (i = 0; i < n3 * n4; i++) B[i] = rand();
		}
		else {
			for (i = 0; i < n1 * n2; i++) A[i] = double_input_validation();
			for (i = 0; i < n3 * n4; i++) B[i] = double_input_validation();
		}
		cout << "How many times colculate for average?" << endl;
		foraverage = natural_input_validation();
	}
	averagetime = 0;
	for (i = 0; i < foraverage; i++) {
		dtime = omp_get_wtime();
		matrix_mult(A, B, C, n1, n2, n4);
		dtime = omp_get_wtime() - dtime;
		averagetime += dtime;
	}
	averagetime /= foraverage;
	printf("%f\n", averagetime);
	if (test) for (i = 0; i < n1*n4; i++) { cout << C[i] << endl; };

	averagetime = 0;
	for (i = 0; i < foraverage; i++) {
		dtime = omp_get_wtime();
		matrix_mult_omp(A, B, C, n1, n2, n4);
		dtime = omp_get_wtime() - dtime;
		averagetime += dtime;
	}
	averagetime /= foraverage;
	printf("%f\n", averagetime);
	if (test) for (i = 0; i < n1 * n4; i++) { cout << C[i] << endl; };

	averagetime = 0;
	for (i = 0; i < foraverage; i++) {
		dtime = omp_get_wtime();
		matrix_mult_block_omp(A, B, C, n1, n2, n4);
		dtime = omp_get_wtime() - dtime;
		averagetime += dtime;
	}
	averagetime /= foraverage;
	printf("%f\n", averagetime);
	if (test) for (i = 0; i < n1 * n4; i++) { cout << C[i] << endl; };

	return 0;
}