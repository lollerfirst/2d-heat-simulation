#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>

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


int main(void)
{
	float buffer[nx * ny * 2] = {0.0f};
	float *points = buffer, *new_points = buffer + (nx * ny);
	
	const float dx_squared = powf(dx, 2);
	const float dy_squared = powf(dy, 2);
	
	size_t x, y;
	
	// Open dat file for the plot
	FILE* f;
	if ((f = fopen(FILENAME, "w")) == NULL)
	{
		perror(strerror(errno));
		return -1;
	}
	
	/*
	// Fill points with 10*(1/(x*y)^3) as an initial state
	size_t y;
	for (y=1; y<ny; ++y)
	{
		size_t x;
		for (x=1; x<nx; ++x)
		{
			points[y*(nx) + x] = (81.0f * powf(((float)x) * ((float)y), -2)) + 19.0f;
			
			if (errno)
			{
				perror(strerror(errno));
				return -1;
			}
			
			points[y * nx] = points[y * nx + 1];
		}
	}
	
	size_t x;
	for (x=0; x<nx; ++x)
	{
		points[x] = points[nx + x];
	}
	*/
	
	// Fill for an initial state with center point at 100 °C 
	for (y=0; y<ny; ++y)
	{
		for (x=0; x<nx; ++x)
		{
			points[y * nx + x] = (x == (nx / 2) && y == (ny / 2)) ? 100.0f : 19.0f;
		}
	}
	
	// print end-line to signal end of frame
	print_graph(f, points);
	fprintf(f, "\n\n");
	
	// print initial points
	printf("Initial State: \n");
	print_points(points);
	
	float t;
	for (t=T_0; t<T_f; t += dt)
	{
		for (y=1; y<ny-1; ++y)
		{
			for (x=1; x<nx-1; ++x)
			{

				new_points[y * nx + x] = points[y * nx + x] + dt * alpha * (
				
					(points[y * nx + x - 1] - (2.0f * points[y * nx + x]) + points[y * nx + x + 1])
							* (1.0f/dx_squared)
					+
					(points[(y-1) * nx + x] - (2.0f * points[y * nx + x]) + points[(y+1) * nx + x])
							* (1.0f/dy_squared)
				);
				
				
			}
			
			// dealing with x boundary
			new_points[y * nx] = new_points[y * nx + 1];
			new_points[y * nx + (nx - 1)] = new_points[y * nx + (nx - 2)];
			
		}
		
		// dealing with y boundary
		for (x=0; x<nx; ++x)
		{
			new_points[x] = new_points[nx + x];
			new_points[(ny-1) * nx + x] = new_points[(ny-2) * nx + x];
			
		}
		
		// print end-line to signal end of frame
		print_graph(f, new_points);
		fprintf(f, "\n\n");
		
		float* temp = points;
		points = new_points;
		new_points = points;
	}
	
	
	fclose(f);
	
	// print ending points
	printf("Ending State: \n");
	print_points(points);
	
	return 0;
}
