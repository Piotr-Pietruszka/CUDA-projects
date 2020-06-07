
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <stdlib.h> 
#include <time.h> 
#include <iomanip>


void print_matrix(float* A, int n);

void print_matrix_code(float* A, int n);

void multiple_matrices(float* A, float* B, float* C, int n);

void add_matrices(float* A, float* B, float* C, int n);

void transpose_matrix(float* A, int n);

void copy_matrix(float* S, float* D, int n);

void normalize_vector(float* A, float* unit_vec_array, int n, int j);

float dot_product(float* A, float* unit_vec_array, int n, int jU, int jA);

void substract_vec(float* A, float* B, float* R, int n, int jA, int jB, int jR, float mn_B = 1.0, float mn_A = 1.0);

void amt_matrices(float* A, float* B, float* C, float* A_T, float* B_T, float* C_T, float* D, int n);

float compare_CPU_GPU(float* A_CPU, float* A_GPU, int n);

float max_error_CPU_GPU(float* A_CPU, float* A_GPU, int n);

float max_element(float* A, int n);
//===========================================================================================

__global__ void add_matrices_GPU(float* A, float* B, float* C, int n)
{

	int id_i = (blockIdx.x * blockDim.x + threadIdx.x);
	int id_j = (blockIdx.y * blockDim.y + threadIdx.y);
	//Dodawanie
	//-----------------------------------
	for (int i = id_i; i < n; i += blockDim.x + gridDim.x)
	{
		for (int j = id_j; j < n; j += blockDim.y + gridDim.y)
		{
			*(C + i * n + j) = (*(A + i * n + j) + *(B + i * n + j));
			//printf("%i, %i, %f\n", i, j, *(A + i * n + j));
		}
	}
	//-----------------------------------
}
__global__ void multiple_matrices_GPU(float* A, float* B, float* C, int n)
{

	int id_i = (blockIdx.x * blockDim.x + threadIdx.x);
	int id_j = (blockIdx.y * blockDim.y + threadIdx.y);
	int stride = blockDim.x * gridDim.x;
	//Mnozenie
	//-----------------------------------
	for (int i = id_i; i < n; i += stride)
	{
		for (int j = id_j; j < n; j += stride)
		{
			float C_ij = 0;
			for (int kAB = 0; kAB < n; kAB++)
			{
				C_ij += (*(A + i * n + kAB)) * (*(B + kAB * n + j));
			}
			*(C + i * n + j) = C_ij;
		}
	}
	//-----------------------------------
}

__global__ void transpose_matrix_GPU(float* A, int n)
{

	//Moze transponowanie z kopiowaniem ??? (zapisywac do innej macierzy - bedzie nawet łatwiej - bez buffa) 
	int id_i = (blockIdx.x * blockDim.x + threadIdx.x);
	int id_j = (blockIdx.y * blockDim.y + threadIdx.y);
	int stride = blockDim.x * gridDim.x;
	float buff = 0;
	//Transponowanie
	//-----------------------------------
	for (int i = id_i; i < n; i += stride)
	{
		for (int j = id_j; j < i; j += stride)
		{
			buff = *(A + i * n + j);
			*(A + i * n + j) = *(A + j * n + i);
			*(A + j * n + i) = buff;
		}
	}
	//-----------------------------------
}

__global__ void transpose_copy_matrix_GPU(float* A, float* A_T, int n)
{

	//Moze transponowanie z kopiowaniem ??? (zapisywac do innej macierzy - bedzie nawet łatwiej - bez buffa) 
	int id_i = (blockIdx.x * blockDim.x + threadIdx.x);
	int id_j = (blockIdx.y * blockDim.y + threadIdx.y);
	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;

	//Transponowanie
	//-----------------------------------
	for (int i = id_i; i < n; i += stride_x)
	{
		for (int j = id_j; j < n; j += stride_y)
		{
			*(A_T + i * n + j) = *(A + j * n + i);
		}
	}
	//-----------------------------------
}

__global__ void add_three_GPU(float* A, float* B, float* C, float* D, int n)
{

	int id_i = (blockIdx.x * blockDim.x + threadIdx.x);
	int id_j = (blockIdx.y * blockDim.y + threadIdx.y);
	//Dodawanie
	//-----------------------------------
	for (int i = id_i; i < n; i += blockDim.x + gridDim.x)
	{
		for (int j = id_j; j < n; j += blockDim.y + gridDim.y)
		{
			*(D + i * n + j) = *(A + i * n + j) + *(B + i * n + j) + *(C + i * n + j);
		}
	}
	//-----------------------------------
}

__global__ void multiple_matrices_shared(float* A, float* B, float* C, int n, const int bl_size)
{
	//Indeksy podbloku macierzy wynikowej C
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	//Indeksy w podbloku
	int row = threadIdx.y;
	int col = threadIdx.x;

	
	for (int i = 0; i < (n / bl_size); i++)
	{
		//__shared__ float As[bl_size][bl_size];
		//__shared__ float Bs[bl_size][bl_size];
	}
}
//===========================================================================================


int main()
{
	bool if_print = false;
	bool if_print_GPU = false;

	srand(time(NULL));

	int n = 1;
	std::cout << "Give n: ";
	std::cin >> n;

	//Allocate memory

	//Macierze dla obliczen na CPU
	float* A = (float*)malloc(n * n * sizeof(float));
	float* B = (float*)malloc(n * n * sizeof(float));
	float* C = (float*)malloc(n * n * sizeof(float));

	float* A_T = (float*)malloc(n * n * sizeof(float));
	float* B_T = (float*)malloc(n * n * sizeof(float));
	float* C_T = (float*)malloc(n * n * sizeof(float));

	float* D = (float*)malloc(n * n * sizeof(float));

	// Initialize with random floats
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			*(A + i * n + j) = (2.0 * rand() / RAND_MAX) - 1.0;
			*(B + i * n + j) = (2.0 * rand() / RAND_MAX) - 1.0;
			
			*(A_T + i * n + j) = *(A + i * n + j);
			*(B_T + i * n + j) = *(B + i * n + j);
		}
	}

	//Obliczenia CPU
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	clock_t start_CPU = clock();

	multiple_matrices(A, B, C, n);
	transpose_matrix(A_T, n);
	transpose_matrix(B_T, n);

	copy_matrix(C, C_T, n);
	transpose_matrix(C_T, n);

	amt_matrices(A, B, C, A_T, B_T, C_T, D, n);



	clock_t stop_CPU = clock();
	//Print
	//====================================
	if (if_print)
	{
		std::cout << "\nA\n";
		print_matrix(A, n);

		std::cout << "\nB\n";
		print_matrix(B, n);

		std::cout << "\nC\n";
		print_matrix(C, n);

		std::cout << "\nA_T\n";
		print_matrix(A_T, n);

		std::cout << "\nD\n";
		print_matrix(D, n);

	}
	//====================================


	std::cout << "Czas_CPU: " << 1000 * (stop_CPU - start_CPU) / ((double)CLOCKS_PER_SEC) << " ms" << std::endl;
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


	//Same GPU
	//=================================================================================================

	//Macierze do skopiowania wynikow GPU
	float* C_h = (float*)malloc(n * n * sizeof(float));
	float* A_T_h = (float*)malloc(n * n * sizeof(float));
	float* B_T_h = (float*)malloc(n * n * sizeof(float));
	float* C_T_h = (float*)malloc(n * n * sizeof(float));
	float* D_h = (float*)malloc(n * n * sizeof(float));


	float* A_dev;
	float* B_dev;
	float* C_dev;

	float* A_T_dev;
	float* B_T_dev;
	float* C_T_dev;

	float* D_dev;
	float* D_A_dev;//Pomocnicze - B*B_T
	float* D_B_dev;//Pomocnicze - B*B_T
	float* D_C_dev;//Pomocnicze - C*C_T

	float* D_AB_dev;//Pomocnicze - B*B_T


	//Alokacja
	cudaMalloc(&A_dev, n * n * sizeof(float));
	cudaMalloc(&B_dev, n * n * sizeof(float));
	cudaMalloc(&C_dev, n * n * sizeof(float));

	cudaMalloc(&A_T_dev, n * n * sizeof(float));
	cudaMalloc(&B_T_dev, n * n * sizeof(float));
	cudaMalloc(&C_T_dev, n * n * sizeof(float));

	cudaMalloc(&D_dev, n * n * sizeof(float));
	cudaMalloc(&D_B_dev, n * n * sizeof(float));
	cudaMalloc(&D_C_dev, n * n * sizeof(float));
	cudaMalloc(&D_A_dev, n * n * sizeof(float));
	cudaMalloc(&D_AB_dev, n * n * sizeof(float));

	//Kopiowanie na Device
	cudaMemcpy(A_dev, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev, B, n * n * sizeof(float), cudaMemcpyHostToDevice);


	int BLOCK_SIZE = 128;
	int GRID_SIZE = 16;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE, GRID_SIZE);

	//obliczenia GPU
	clock_t start_GPU = clock();
	//C
	multiple_matrices_GPU <<<dimBlock, dimGrid >>> (A_dev, B_dev, C_dev, n);
	//Transpozycje
	transpose_copy_matrix_GPU <<<dimBlock, dimGrid >>> (A_dev, A_T_dev, n);
	transpose_copy_matrix_GPU <<<dimBlock, dimGrid >>> (B_dev, B_T_dev, n);
	transpose_copy_matrix_GPU <<<dimBlock, dimGrid >>> (C_dev, C_T_dev, n);
	
	// D matrix
	cudaDeviceSynchronize();
	multiple_matrices_GPU <<<dimBlock, dimGrid >>> (A_dev, A_T_dev, D_A_dev, n);
	cudaDeviceSynchronize();

	multiple_matrices_GPU <<<dimBlock, dimGrid >>> (B_dev, B_T_dev, D_B_dev, n);
	cudaDeviceSynchronize();

	multiple_matrices_GPU <<<dimBlock, dimGrid >>> (C_dev, C_T_dev, D_C_dev, n);
	cudaDeviceSynchronize();


	add_three_GPU <<<dimBlock, dimGrid >>> (D_A_dev, D_B_dev, D_C_dev, D_dev, n);

	cudaDeviceSynchronize();


	//Kopiowanie na Hosta
	cudaMemcpy(C_h, C_dev, n * n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(A_T_h, A_T_dev, n * n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(D_h, D_dev, n * n * sizeof(float), cudaMemcpyDeviceToHost);

	//Stop 
	clock_t stop_GPU = clock();


	if (if_print_GPU)
	{
		std::cout << "\nA\n";
		print_matrix(A, n);

		std::cout << "\nB\n";
		print_matrix(B, n);

		std::cout << "\nC_h\n";
		print_matrix(C_h, n);

		std::cout << "\nA_T_h\n";
		print_matrix(A_T_h, n);

		std::cout << "\nD_h\n";
		print_matrix(D_h, n);

	}

	//Porownanie GPU i CPU
	std::cout << "Czas_GPU: " << 1000 * (stop_GPU - start_GPU) / ((double)CLOCKS_PER_SEC) << " ms" << std::endl;
	float C_error = compare_CPU_GPU(A_T, A_T_h, n);
	//std::cout << "D errory: " << std::endl;
	float D_error = compare_CPU_GPU(D, D_h, n);
	float Max_D_error = max_error_CPU_GPU(D, D_h, n);


	std::cout << "Blad w C: " << C_error << std::endl;
	std::cout << "Blad w D: " << D_error << std::endl;
	std::cout << "Maksymalny blad w D: " << Max_D_error << std::endl;


	if (stop_GPU - start_GPU > 0)
	{
		std::cout << "Speedup: " << 1.0 * (stop_CPU - start_CPU) / (stop_GPU - start_GPU) << std::endl;
	}
	else
	{
		std::cout << "Czas obliczen zbyt krotki by okreslic speedup" << std::endl;
	}
	//=================================================================================================
	
	//Zwolnienie pamieci macierzy od GPU
	free(C_h);
	free(A_T_h);
	free(B_T_h);
	free(C_T_h);
	free(D_h);

	//Zwolnienie pamieci  GPU
	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);
	cudaFree(A_T_dev);
	cudaFree(B_T_dev);
	cudaFree(C_T_dev);
	cudaFree(D_dev);

	//Zwolnienie pamieci macierzy od CPU
	free(A);
	free(B);
	free(C);
	free(A_T);
	free(B_T);
	free(C_T);
	free(D);


	return 0;
}




//##################################################################################3
float compare_CPU_GPU(float* A_CPU, float* A_GPU, int n)
{
	float error = 0.0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			error += powf(*(A_CPU + i * n + j) - *(A_GPU + i * n + j), 2.0);
		}
	}
	return error;
}

float max_error_CPU_GPU(float* A_CPU, float* A_GPU, int n)
{
	float max_error = 0.0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float error = powf(*(A_CPU + i * n + j) - *(A_GPU + i * n + j), 2.0);
			if (error > max_error)
			{
				//std::cout << "*(A_CPU + i * n + j): " << *(A_CPU + i * n + j) << "\n *(A_GPU + i * n + j): " << *(A_GPU + i * n + j) << "\n\n";
				max_error = error;
			}

		}
	}
	return max_error;
}
float max_element(float* A, int n)
{
	float max_el = 0.0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (*(A + i * n + j) > max_el)
			{
				//std::cout << "*(A + i * n + j): " << *(A + i * n + j) << "\n\n";
				max_el = *(A + i * n + j);
			}

		}
	}
	return max_el;
}


void print_matrix(float* A, int n)
{
	for (int i = 0; i < n; i++)
	{

		for (int j = 0; j < n; j++)
		{
			std::cout << std::setw(12) << *(A + i * n + j) << " ";
		}
		std::cout << "\n";
	}
}

void print_matrix_code(float* A, int n)
{
	//WYPISANIE KOLUMNAMI
	std::cout << "[";
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			std::cout << *(A + j * n + i);
			if (j != n - 1)
				std::cout << ",";
		}
		if (i != n - 1)
			std::cout << "; ";
	}
	std::cout << "]";
}

void multiple_matrices(float* A, float* B, float* C, int n)
{
	//Mnozenie
	//-----------------------------------
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			//std::cout << "\n\n" << i << ", " << j << ": \n";
			float C_ij = 0;
			for (int kAB = 0; kAB < n; kAB++)
			{
				C_ij += (*(A + i * n + kAB)) * (*(B + kAB * n + j));
				//std::cout << "A " << i << ", " << kAB << "    *B " << kAB << ", " << j <<  "    : " << C_ij << "\n";				
			}
			*(C + i * n + j) = C_ij;
		}
	}
	//-----------------------------------
}

void add_matrices(float* A, float* B, float* C, int n)
{
	//Dodawanie
	//-----------------------------------
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			*(C + i * n + j) = (*(A + i * n + j) + *(B + i * n + j));
		}
	}
	//-----------------------------------
}

void transpose_matrix(float* A, int n)
{
	//Transponowanie
	//-----------------------------------
	float buff = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < i; j++)
		{
			buff = *(A + i * n + j);
			*(A + i * n + j) = *(A + j * n + i);
			*(A + j * n + i) = buff;
		}
	}
	//-----------------------------------
}

void copy_matrix(float* S, float* D, int n)
{
	//Kopiowanie
	//-----------------------------------
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			*(D + i * n + j) = *(S + i * n + j);
		}
	}
	//-----------------------------------
}

void normalize_vector(float* A, float* unit_vec_array, int n, int j)
{
	//Normalizing column vector (matrix A, col j) 
	//-----------------------------------
	float norm = 0.0;
	for (int i = 0; i < n; i++)//calculate norm
	{
		norm += pow(*(A + i * n + j), 2.0);
	}
	norm = sqrt(norm);

	for (int i = 0; i < n; i++)//calculate unit vector
	{
		*(unit_vec_array + i * n + j) = *(A + i * n + j) / norm;
	}
	//-----------------------------------
}

float dot_product(float* A, float* unit_vec_array, int n, int jU, int jA)
{
	float d_p = 0;
	//Iloczyn skalarny
	//-----------------------------------

	for (int i = 0; i < n; i++)//calculate unit vector
	{
		d_p += (*(unit_vec_array + i * n + jU)) * (*(A + i * n + jA));
		//std::cout << "i: " << i << ",  u_vec_a: = " << *(unit_vec_array + i * n + jU) << ", A_element = " << *(A + i * n + jA) << ",  d_p: " << d_p << "\n";
	}
	return d_p;
	//-----------------------------------
}

void substract_vec(float* A, float* B, float* R, int n, int jA, int jB, int jR, float mn_B, float mn_A)//mn - mnoznik
{
	float d_p = 0;
	//Odejmowanie
	//-----------------------------------
	for (int i = 0; i < n; i++)//calculate unit vector
	{
		//std::cout << "i: " << i << ", A_el = " << *(A + i * n + jA) << "\n";
		*(R + i * n + jR) = mn_A * (*(A + i * n + jA)) - mn_B * (*(B + i * n + jB));
		//std::cout << "i: "<< i << ",  R_el = " << *(R + i * n + jR) << ", A_el = " << *(A + i * n + jA) << ",  B_el =   " << *(B + i * n + jB) 
		//	<< ", B_el_multiplied = " << mn_B * (*(B + i * n + jB)) <<  "\n";
	}
	//-----------------------------------
}

void amt_matrices(float* A, float* B, float* C, float* A_T, float* B_T, float* C_T, float* D, int n)
{
	//D = A*A_T + B*B_T + C*C_T
	float* AA_T = (float*)malloc(n * n * sizeof(float));
	float* BB_T = (float*)malloc(n * n * sizeof(float));
	float* CC_T = (float*)malloc(n * n * sizeof(float));

	multiple_matrices(A, A_T, AA_T, n);
	multiple_matrices(B, B_T, BB_T, n);
	multiple_matrices(C, C_T, CC_T, n);
	add_matrices(AA_T, BB_T, D, n);
	add_matrices(D, CC_T, D, n);

	free(AA_T);
	free(BB_T);
	free(CC_T);
}

