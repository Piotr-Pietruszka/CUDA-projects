#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <stdlib.h> 
#include <time.h> 
#include <iomanip>
#include <math.h>

using namespace std;
typedef double myfloat;

const double pi_const = 3.1415926535897932384626433832795;

void generate_components(float* pi_comp, int n);
float add_components(float* pi_comp, int n);
void print_components(float* pi_comp, int n);

void generate_components(double* pi_comp, int n);
double add_components(double* pi_comp, int n);
void print_components(double* pi_comp, int n);



__global__ void add_components_GPU_2(myfloat* pi_components, myfloat* pi_components_2, int thr_adds)
{
	int id_i = (blockIdx.x * blockDim.x + threadIdx.x) * thr_adds;
	for (int i = id_i; i < id_i + thr_adds; i++)
	{
		*(pi_components_2 + id_i / thr_adds) += *(pi_components + i);
	}
}

__global__ void generate_GPU(myfloat* pi_components, int n)
{
	int id_i = (blockIdx.x * blockDim.x + threadIdx.x);

	for (int i = id_i; i < n; i += blockDim.x + gridDim.x)
	{
		*(pi_components + i) = 4 * 1.0 / (2 * i + 1) * ((2 * i) % 4 ? -1.0 : 1.0);
	}
}



int main()
{
	int n = 1;
	//std::cout << "Give n: ";
	//std::cin >> n;
	n = 600000000;

	std::cout << "n: " << n << "\n\n";

	myfloat* pi_comp_CPU = (myfloat*)malloc(n * sizeof(myfloat));
	cudaError_t cudaStatus;
//Obliczenia CPU
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	generate_components(pi_comp_CPU, n);
	clock_t start_CPU = clock();

	myfloat pi_no = add_components(pi_comp_CPU, n);

	clock_t stop_CPU = clock();
	std::cout << "pi_constant: " << std::setprecision(50) << pi_const << "\n\n";
	std::cout << "\n\nCPU\n";
	std::cout << "pi CPU: " << std::setprecision(50) << pi_no << "\n";
	std::cout << "Czas_CPU: " << 1000 * (stop_CPU - start_CPU) / ((double)CLOCKS_PER_SEC) << " ms" << std::endl;

	std::cout << "pi_error: " << std::setprecision(50) << pi_const - pi_no << "\n\n";
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


	//GPU
	myfloat* pi_components;
	myfloat* pi_components_2;

	int BLOCK_SIZE = 256;
	int GRID_SIZE = 16;

	myfloat* pi_sum_h = (myfloat*)malloc(sizeof(myfloat));
	cudaStatus = cudaMalloc(&pi_components, (n + BLOCK_SIZE * GRID_SIZE) * sizeof(myfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
	}
	cudaStatus = cudaMalloc(&pi_components_2, BLOCK_SIZE * GRID_SIZE * sizeof(myfloat));//(n+1)/2 
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
	}


	

	cudaMemcpy(pi_components, pi_comp_CPU, n * sizeof(myfloat), cudaMemcpyHostToDevice);
	clock_t start_GPU = clock();
// Obliczenia GPU
//--------------------------------------------------------------
	int thr_adds = (n + BLOCK_SIZE * GRID_SIZE - 1) / (BLOCK_SIZE * GRID_SIZE);//ile jeden watek ma wykonac dodawan
	add_components_GPU_2 <<<GRID_SIZE, BLOCK_SIZE >>> (pi_components, pi_components_2, thr_adds);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "add_components_GPU_2 launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaMemcpy(pi_comp_CPU, pi_components_2, (BLOCK_SIZE * GRID_SIZE) * sizeof(myfloat), cudaMemcpyDeviceToHost);
	*pi_sum_h = add_components(pi_comp_CPU, (BLOCK_SIZE * GRID_SIZE));
//--------------------------------------------------------------

	clock_t stop_GPU = clock();
	std::cout << "\n\nGPU\n";
	std::cout << "pi GPU: " << std::setprecision(50) << *pi_sum_h << "\n";
	std::cout << "Czas_GPU: " << 1000 * (stop_GPU - start_GPU) / ((double)CLOCKS_PER_SEC) << " ms" << std::endl;
	std::cout << "pi_error: " << std::setprecision(50) << pi_const - *pi_sum_h << "\n\n";

	std::cout << "Speedup: " << (double)(stop_CPU - start_CPU) / (stop_GPU - start_GPU) << "\n\n";

	cudaFree(pi_components);
	cudaFree(pi_components_2);
	free(pi_comp_CPU);
	free(pi_sum_h);
	return 0;
}


//##################################################################################
void generate_components(myfloat* pi_comp, int n)
{
	int sign = 1;
	for (int i = 1; i < 2 * n; i += 2)
	{
		*(pi_comp + i / 2) = 4 * 1.0 / i * sign;
		sign = -sign;
	}
}

myfloat add_components(myfloat* pi_comp, int n)
{
	myfloat pi_no = 0.0;

	for (int i = 0; i < n; i++)
	{
		pi_no += pi_comp[i];
	}
	return pi_no;
}

void print_components(myfloat* pi_comp, int n)
{
	for (int i = 0; i < n; i++)
	{
		std::cout << "comp " << i << ": " << pi_comp[i] << "\n";
	}
}


