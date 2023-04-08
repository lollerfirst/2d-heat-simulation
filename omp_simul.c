#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <omp.h>

// Max iters
#define MAX_ITERS 500
#define CHECKPOINT 100

// Length of the domain
#define Lx 30
#define Ly 30

// Number of points
#define nx (Lx + 1)
#define ny (Ly + 1)

// Discretized differentials
#define dx 1.0f
#define dy 1.0f
#define dt 0.01f

// Start, end time
#define T_0 0
#define T_f 3

// Thermal diffusivity
#define alpha 0.5

// Graph filenames
#define FILENAME "heat_diffusion.dat"

void print_graph(FILE* f, const float* points)
{
	size_t y,x;
	
	for (y=0; y<ny; ++y)
	{
		for (x=0; x<nx; ++x)
		{
			fprintf(f, "%lu %lu %.3f\n", x, y, points[y * nx + x]);
		}
	}
}

void print_points(const float* points)
{
	size_t y;
	size_t x;
	
	for (y=0; y<ny; ++y)
	{
		for (x=0; x<nx; ++x)
		{
			printf("%.4f ", points[y * nx + x]);
		}
		printf("\n");
	}
}

void checkpoint(FILE* f, const float* points, size_t process_rank)
{
	static size_t c = 0;
	
	printf("[%lu] Checkpoint %lu\n", process_rank, ++c);
	const float* temp = points - (CHECKPOINT-1) * nx * ny;

	for (size_t i = 0; i < CHECKPOINT; ++i)
	{
		print_graph(f, temp + i * ny * nx);
		fprintf(f, "\n\n");
	}
}


int main(int argc, char** argv)
{
	// Allocate buffer
	float *buffer = (float*) calloc(nx * ny * (CHECKPOINT+1), sizeof(float));
	
	if (buffer == NULL)
	{
		perror(strerror(errno));
		return -1;
	}
	
	// Declare pointers
	float *points, *new_points;
	
	// Initialize spatial differentials
	const float dx_squared = powf(dx, 2);
	const float dy_squared = powf(dy, 2);
	
	size_t x, y;
	
	// Open file to store the results in
	FILE* f;
	if ((f = fopen(FILENAME, "w")) == NULL)
	{
		perror(strerror(errno));
		free(buffer);
		return -1;
	}
	
	printf("Cores: %d\n", omp_get_max_threads());
	
	
	// Fill for an initial state with center point at 100 °C 
	for (y=0; y<ny; ++y)
	{
		for (x=0; x<nx; ++x)
		{
			buffer[y * nx + x] = (x == (nx / 2) && y == (ny / 2)) ? 100.0f : 19.0f;
		}
	}
	
	
	/*
	// print initial points
	printf("Initial State: \n");
	print_points(points);
	*/
	
	
	points = buffer;
	new_points = points + ny * nx;

	// checkpoint counter
	size_t c = 0;
	
	// timestep variable
	size_t t;

	for (t = 1; t<MAX_ITERS; ++t)
	{
		
		// Checkpointing procedure
		if (t % CHECKPOINT == 0)
		{
			checkpoint(f, new_points, 0);
		}
				
		// Adjust pointers to next frame
		points = buffer + ((t-1) % CHECKPOINT) * ny * nx;
		new_points = buffer + (t % CHECKPOINT) * ny * nx;
		
		
		// Grid points computation
		#pragma omp parallel for
		for (size_t i=0; i<(ny*nx); ++i)
		{
			//float stencil[4] =  {0};
		
			y = i / nx;
			x = i % nx;
			
			
			// Enumerating boundary cases
			int jmp = 0;
			jmp += (x == 0 && y == 0);
			jmp += (x == 0 && y == (ny-1)) << 1;
			jmp += (x == (nx-1) && y == 0) << 2;
			jmp += (x == (nx-1) && y == (ny-1)) << 3;
			
			jmp += (x == 0 && y != 0 && y != (ny-1)) << 4;
			jmp += (y == 0 && x != 0 && x != (nx-1)) << 5;
			jmp += (x == (nx-1) && y != 0 && y != (ny-1)) << 6;
			jmp += (y == (ny-1) && x != 0 && x != (nx-1)) << 7;
			
			jmp += (x != 0 && y != (ny-1) && x != (nx-1) && y != 0) << 8;
			
			/* If on a boundary, compute the average of the neighboring cells.
			   If on a internal cell, compute the FTCS (Forward in time, Central in space) */
			     
			switch(jmp)
			{
			// 1
			case 1:
				new_points[0] = (points[1] / 2.0f + points[nx] / 2.0f);
				break;
			// 2
			case 2:
				new_points[y * nx] = (points[y * nx + 1] / 2.0f + points[(y-1) * nx] / 2.0f);
				break;
			// 3
			case 4:
				new_points[x] = (points[x-1] / 2.0f + points[nx + x] / 2.0f);
				break;
			// 4
			case 8:
				new_points[y * nx + x] = (points[y * nx + x-1] / 2.0f + points[(y-1) * nx + x] / 2.0f);
				break;
			// 5
			case 16:
				new_points[y * nx] = points[y * nx + 1] / 3.0f + points[(y-1) * nx] / 3.0f
					+ points[(y+1) * nx] / 3.0f;
				break;
			// 6
			case 32:
				new_points[x] = points[x-1] / 3.0f + points[nx + x] / 3.0f
					+ points[x+1] / 3.0f;
				break;
			
			// 7
			case 64:
				new_points[y * nx + x] = points[(y-1) * nx + x] / 3.0f + points[y * nx + x-1] / 3.0f
					+ points[(y+1) * nx + x] / 3.0f;
				break;
			
			// 8
			case 128:
				new_points[y * nx + x] = points[y * nx + x-1] / 3.0f + points[(y-1) * nx + x] / 3.0f
					+ points[y * nx + x+1] / 3.0f;
				break;
				
			// 9	
			case 256:
				
				//get_neighboring_temperatures(stencil);
				new_points[y * nx + x] = points[y * nx + x] + dt * alpha * (
				
					(points[y * nx + x - 1] - (2.0f * points[y * nx + x]) + points[y * nx + x + 1])
							* (1.0f/dx_squared)
					+
					(points[(y-1) * nx + x] - (2.0f * points[y * nx + x]) + points[(y+1) * nx + x])
							* (1.0f/dy_squared)
				);
			default:
				break;
			}
			
			// Maintain midpoint to 100°C to simulate a laser application heat source
			if (x == (nx/2) && y == (ny / 2))
			{
				new_points[y * nx + x] = points[y * nx + x];
			}
		
		}
		
	}
	
	// Last checkpoint
	checkpoint(f, new_points, 0);
	
	fclose(f);
	free(buffer);
	
	/*
	// print ending points
	printf("Ending State: \n");
	print_points(points);
	*/
	
	return 0;
}
