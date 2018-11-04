#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#define INT_DECIMAL_STRING_SIZE(int_type) ((CHAR_BIT*sizeof(int_type)-1)*10/33+3)


using namespace std;

#define blocksize 32

void display_vector(vector<int> v)
{
	for (int i = 0; i < (int)v.size(); i++)
		cout<< v.at(i) <<" ";
}
void display_vector(vector<float> v)
{
	for (int i = 0; i < (int)v.size(); i++)
		cout<< v.at(i) <<" ";
}

/*storing matrix*/
void matrix_read(string filename,float *L, int dimension){

	FILE *fp;
	int row, col;

	fp = fopen(filename.c_str(), "r");//open output file
	if (fp == NULL)//open failed
		return;

	for (row = 0; row < dimension; row++){
		for (col = 0; col < dimension; col++)
		if (fscanf(fp, "%f,", &L[row * dimension + col]) == EOF) break;//read data

		if (feof(fp)) break;//if the file is over
	}

	fclose(fp);//close file

}

__global__ void nodiag_normalize(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == i && x!=y){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}
	
}

__global__ void diag_normalize(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == y && x == i){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}

}

__global__ void gaussjordan(float *A, float *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i){
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}	 
		}
	}

}

__global__ void set_zero(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			if (y == i){
				A[x*n + y] = 0;
			}
		}
	}
}

void savetofile(float *A, string s, int n, int h)
{
	std::ofstream plik;
	plik.open(s);

	for (int j = 0; j<h; j++){
		for (int i = 0; i<h; i++){
			plik << A[j*n + i] << "\t";
		}
		plik << endl;
	}
	plik.close();
}
float execute(string filename,int size){
	const int n = size;
	// creating input
	float *iL = new float[n*n];
	float *L = new float[n*n];

	matrix_read(filename,L, n);
	//savetofile(L, "L.txt", n, n);

	float *d_A, *I, *dI;
	float time;
	cudaError_t err;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int ddsize = n*n*sizeof(float);

	dim3 threadsPerBlock(blocksize, blocksize);
	dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
	// memory allocation    
	err = cudaMalloc((void**)&d_A, ddsize);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	err = cudaMalloc((void**)&dI, ddsize);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	I = new float[n*n];

	for (int i = 0; i<n; i++){
		for (int j = 0; j<n; j++){
			if (i == j) I[i*n + i] = 1.0;
			else I[i*n + j] = 0.0;
		}
	}

	//copy data from CPU to GPU
	err = cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	err = cudaMemcpy(dI, I, ddsize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }

	//timer start
	cudaEventRecord(start, 0);

	// L^(-1)    
	for (int i = 0; i<n; i++){
		nodiag_normalize << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		diag_normalize << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		gaussjordan << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		set_zero << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copy data from GPU to CPU
	err = cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	err = cudaMemcpy(I, d_A, ddsize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }

	savetofile(iL, "inv.txt", n, n);
	//savetofile(I, "I.txt", n, n);
	cudaFree(d_A);
	cudaFree(dI);

	float *c = new float[n*n];
	for (int i = 0; i<n; i++)  
	for (int j = 0; j<n; j++)  
	{
		c[i*n+j] = 0;  //put the initial value to zero
		for (int x = 0; x<n; x++)  
			c[i*n + j] = c[i*n + j] + L[i*n+x] * iL[x*n + j];  //matrix multiplication
	}
	savetofile(c, "c.txt", n, n);

	delete[]I;
	delete[]L;
	delete[]iL;
	return time;
}

char *stringer(int x) {
  int i = x;
  char buf[INT_DECIMAL_STRING_SIZE(int)];
  char *p = &buf[sizeof buf - 1];
  *p = '\0';
  if (i >= 0) {
    i = -i;
  }
  do {
    p--;
    *p = (char) ('0' - i % 10);
    i /= 10;
  } while (i);
  if (x < 0) {
    p--;
    *p = '-';
  }
  size_t len = (size_t) (&buf[sizeof buf] - p);
  char *s = (char*)malloc(len);
  if (s) {
    memcpy(s, p, len);
  }
  return s;
}

int main()
{

	char matrix_types[100][100]={"dense","sparse","hollow","band","identity"};
	execute("input/dense/1000.txt",1000);
	cout<<"\n\nGauss Jordan Inversion GPU Implementation\n\n";
	for(int j=0;j<=4;j++){
		std::vector<float> time;
		std::vector<int> ns;

		for(int i=50;i<=1000;i+=50){
			string num (stringer(i));
			string root ("input/");
			string type (matrix_types[j]);
			string ext (".txt");
			string dir (root+type+"/"+num+ext);
			float t=execute(dir,i);
			ns.push_back(i);
			time.push_back(t);
			cout<<type<<" matrix of size "<<i<<" took "<<t<<" ms\n";
		}
		display_vector(ns);
		cout<<endl;
		display_vector(time);		
		cout<<endl;
	}
}