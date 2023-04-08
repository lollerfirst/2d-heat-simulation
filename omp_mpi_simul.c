#include <stdio.h>
#include <cmath>
#include <errno.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

// Max iters
#define MAX_ITERS 500
#define CHECKPOINT 100

// Length of the domain
#define Lx 30
#define Ly 30

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

static size_t nx,ny;
static int neighbors[4];
static char estring[MPI_MAX_ERROR_STRING] = {0};

enum directions
{
	NORTH,
	WEST,
	SOUTH,
	EAST
};

enum tags
{
	FRAME_INIT,
	REQ_VALUE,
	SUPPL_VALUE
};

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

template <size_t DIRECTION>
int get_remote_value(size_t x, float* out)
{
	size_t arr[3] = {-1, nx, ny};
	int err;
	
	
	if constexpr (DIRECTION == NORTH)
	{
		// SOUTH of neighbor
		arr[0] = (ny-1)*nx + x;
		
		#pragma omp critical(critical_north)
		{
			err = MPI_Send(arr, 3, MPI_UNSIGNED_LONG, neighbors[NORTH], REQ_VALUE, MPI_COMM_WORLD);
			
			if (err)
			{
				MPI_Error_string(err, estring, nullptr);
				fprintf(stderr, "MPI ERROR: %s\n", estring);
				//MPI_Finalize();
				return err;
			}
			
			err = MPI_Recv(out, 1, MPI_FLOAT, neighbors[NORTH], SUPPL_VALUE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			if (err)
			{
				MPI_Error_string(err, estring, nullptr);
				fprintf(stderr, "MPI ERROR: %s\n", estring);
				//MPI_Finalize();
				return err;
			}
			
			
		}

	}
	
	if constexpr (DIRECTION == WEST)
	{
		// EAST of neighbor
		arr[0] = x * nx + nx-1;
		
		#pragma omp critical(critical_west)
		{
			err = MPI_Send(arr, 3, MPI_UNSIGNED_LONG, neighbors[WEST], REQ_VALUE, MPI_COMM_WORLD);
			
			if (err)
			{
				MPI_Error_string(err, estring, nullptr);
				fprintf(stderr, "MPI ERROR: %s\n", estring);
				
				return err;
			}
			
			err = MPI_Recv(out, 1, MPI_FLOAT, neighbors[WEST], SUPPL_VALUE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			if (err)
			{
				MPI_Error_string(err, estring, nullptr);
				fprintf(stderr, "MPI ERROR: %s\n", estring);
				
				return err;
			}
		}
	
	}
	
	if constexpr (DIRECTION == SOUTH)
	{
		// NORTH of neighbor
		arr[0] = x;
		
		
		#pragma omp critical(critical_south)
		{
			err = MPI_Send(arr, 3, MPI_UNSIGNED_LONG, neighbors[SOUTH], REQ_VALUE, MPI_COMM_WORLD);
			
			if (err)
			{
				MPI_Error_string(err, estring, nullptr);
				fprintf(stderr, "MPI ERROR: %s\n", estring);
				
				return err;
			}
			
			err = MPI_Recv(out, 1, MPI_FLOAT, neighbors[SOUTH], SUPPL_VALUE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			if (err)
			{
				MPI_Error_string(err, estring, nullptr);
				fprintf(stderr, "MPI ERROR: %s\n", estring);
				
				return err;
			}
		}
	}
	
	if constexpr (DIRECTION == EAST)
	{
		// WEST of neighbor
		arr[0] = x*nx;
		
		
		#pragma omp critical(critical_east)
		{
			err = MPI_Send(arr, 3, MPI_UNSIGNED_LONG, neighbors[EAST], REQ_VALUE, MPI_COMM_WORLD);
			
			if (err)
			{
				MPI_Error_string(err, estring, nullptr);
				fprintf(stderr, "MPI ERROR: %s\n", estring);
				
				return err;
			}
			
			err = MPI_Recv(out, 1, MPI_FLOAT, neighbors[EAST], SUPPL_VALUE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			if (err)
			{
				MPI_Error_string(err, estring, nullptr);
				fprintf(stderr, "MPI ERROR: %s\n", estring);
				
				return err;
			}
		}
	}
	
	return 0;
}


int get_neighboring_values(size_t x, size_t y, const float *points, float *out_values)
{
	int err;
	
	if (x == 0 && y == 0)
	{
		out_values[EAST] = points[y*nx + x + 1];
		out_values[SOUTH] = points[(y+1)*nx + x];
		
		if (neighbors[NORTH] != -1)
		{
			err = get_remote_value<NORTH>(x, &out_values[NORTH]);
			
			if (err)
			{
				return -1;
			}
		
			if (neighbors[WEST] != -1)
			{
				err = get_remote_value<WEST>(y, &out_values[WEST]);
				if (err)
				{
					return -1;
				}
				
				return 1 << 8;
			}
			
			return 1 << 4;
		}
		
		return 1;
	}
	
	if  (x == 0 && y == (ny-1))
	{
		out_values[EAST] = points[y*nx + x + 1];
		out_values[NORTH] = points[(y-1)*nx + x];
		
		if (neighbors[SOUTH] != -1)
		{
			err = get_remote_value<SOUTH>(x, &out_values[SOUTH]);
			
			if (err)
			{
				return -1;
			}
		
			if (neighbors[WEST] != -1)
			{
				err = get_remote_value<WEST>(y, &out_values[WEST]);
				if (err)
				{
					return -1;
				}
				
				return 1 << 8;
			}
			
			return 1 << 4;
		}
		
		return 1 << 1;
	}
	
	if (x == (nx-1) && y == 0)
	{
		
	}
	jmp += (x == (nx-1) && y == (ny-1)) << 3;
	
	jmp += (x == 0 && y != 0 && y != (ny-1)) << 4;
	jmp += (y == 0 && x != 0 && x != (nx-1)) << 5;
	jmp += (x == (nx-1) && y != 0 && y != (ny-1)) << 6;
	jmp += (y == (ny-1) && x != 0 && x != (nx-1)) << 7;
	
	jmp += (x != 0 && y != (ny-1) && x != (nx-1) && y != 0) << 8;
}


int main(int argc, char** argv)
{
	// Error variables
	int err;

	// Initialize MPI
	err = MPI_Init(&argc, &argv);
	if (err)
	{
		MPI_Error_string(err, estring, nullptr);
		fprintf(stderr, "MPI ERROR: %s\n", estring);
		return err;
	}
	
	// Get the total number of nodes
	int n_nodes;
	err = MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
	
	if (err)
	{
		MPI_Error_string(err, estring, nullptr);
		fprintf(stderr, "MPI ERROR: %s\n", estring);
		MPI_Finalize();
		return err;
	}
	
	// Get the rank of this process
	int rank;
	err = MPI_Comm_rank(MPI_COMM_WORLD, &rank)
	
	if (err)
	{
		MPI_Error_string(err, estring, nullptr);
		fprintf(stderr, "MPI ERROR: %s\n", estring);
		MPI_Abort(MPI_COMM_WORLD, err);
		MPI_Finalize();
		return err;
	}
	
	// Calculate size of chunk
	size_t sqrt_n_nodes_x = (size_t) std::floor(std::sqrt((float)n_nodes));
	size_t sqrt_n_nodes_y = (size_t) std::ceil(std::sqrt((float)n_nodes));
	size_t full_ny, full_nx;
	
	// Allocate memory and scatter domain from master to slaves
	float *buffer = nullptr;
	float *recv_buffer = nullptr;
	
	if (rank == 0)
	{
		full_ny = Ly + 1;
		full_nx = Lx + 1;
		
		buffer = (float*) calloc(full_ny * full_nx * (CHECKPOINT+1), sizeof(float));
		
		if (buffer == nullptr)
		{
			MPI_Error_string(err, estring, nullptr);
			fprintf(stderr, "MPI ERROR: %s\n", estring);
			MPI_Abort(MPI_COMM_WORLD, err);
			MPI_Finalize();
			return err;
		}
		
		// Fill for an initial state with center point at 100 °C 
		for (y=0; y<full_ny; ++y)
		{
			for (x=0; x<full_nx; ++x)
			{
				buffer[y * full_nx + x] = (x == (full_nx / 2) && y == (full_ny / 2)) ? 100.0f : 19.0f;
			}
		}
	
	}
	
	nx = (Lx + 1) / sqrt_n_nodes_x;
	ny = (Ly + 1) / sqrt_n_nodes_y;
	
	size_t rem_nx = (Lx + 1) % sqrt_n_nodes_x;
	size_t rem_ny = (Ly + 1) % sqrt_n_nodes_y;
	
	
	recv_buffer = (float*) calloc((nx+rem_nx) * (ny+rem_ny) * (CHECKPOINT+1), sizeof(float));
	
	if (recv_buffer == nullptr)
	{
		MPI_Error_string(err, estring, nullptr);
		fprintf(stderr, "MPI ERROR: %s\n", estring);
		MPI_Abort(MPI_COMM_WORLD, err);
		MPI_Finalize();
		free(buffer);
		return err;
	}
	
	// Distribute initial frames amongst nodes	
	if (rank == 0)
	{
		for (size_t i=0; i<full_ny; ++i)
		{
			float *starting_ptr;
			int to;
			size_t count;
			
			for (size_t j=0; j<sqrt_n_nodes_x; ++j)
			{
				starting_ptr = buffer + i*full_nx + j*nx;
				to = (i/ny)*sqrt_n_nodes_x + j;
				count = nx + ((j == sqrt_n_nodes_x-1) ? rem_nx : 0);
				
				err = MPI_Send(starting_ptr, count, MPI_FLOAT, to, MSG_INIT, MPI_COMM_WORLD);
				
				if (err)
				{
					MPI_Error_string(err, estring, nullptr);
					fprintf(stderr, "MPI ERROR: %s\n", estring);
					MPI_Abort(MPI_COMM_WORLD, err);
					MPI_Finalize();
					free(buffer);
					free(recv_buffer);
					return err;
				}
			}
		}
	}
	
	// Receive initial frame from master node
	nx += ((rank+1) % sqrt_n_nodes_x == 0) ? rem_nx : 0;
	ny += (rank+1 - (sqrt_n_nodes_y-1) * sqrt_n_nodes_x >= 0) ? rem_ny : 0;
	
	size_t recv_size = nx * ny;
	
	err = MPI_Recv(recv_buffer, recv_size, MPI_FLOAT, 0, MSG_INIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	if (err)
	{
		MPI_Error_string(err, estring, nullptr);
		fprintf(stderr, "MPI ERROR: %s\n", estring);
		MPI_Abort(MPI_COMM_WORLD, err);
		MPI_Finalize();
		free(buffer);
		free(recv_buffer);
		return err;
	}
	
	
	// Declare pointers
	float *points, *new_points;
	
	// Initialize spatial differentials
	const float dx_squared = std::powf(dx, 2);
	const float dy_squared = std::powf(dy, 2);
	
	// Gridpoint variables
	size_t x, y;
	
	// Open file to store the results in
	FILE* f;
	if ((f = fopen(FILENAME, "w")) == nullptr)
	{
		perror(strerror(errno));
		free(buffer);
		free(recv_buffer);
		MPI_Finalize();
		return -1;
	}
	
	printf("Cores: %d\n", omp_get_max_threads());
	
	

	// NORTH
	neighbors[NORTH] = (rank >= sqrt_n_nodes_x) ? (rank - sqrt_n_nodes_x - 1) : -1;
	
	// WEST
	neighbors[WEST] = (rank % sqrt_n_nodes_x != 0) ? (rank - 1) : -1;
	
	// SOUTH
	neighbors[SOUTH] = (rank < (sqrt_n_nodes_y-1) * (sqrt_n_nodes_x)) ? (rank + sqrt_n_nodes_x) : -1;
	
	// EAST
	neighbors[EAST] = ((rank+1) % sqrt_n_nodes_x != 0) ? (rank + 1) : -1;
		
	/*
	// print initial points
	printf("Initial State: \n");
	print_points(points);
	*/
	
	// Pointers to frame
	points = recv_buffer;
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
					
			y = i / nx;
			x = i % nx;
			
			float values[4];
			int jmp = 0;
			jmp = get_neighboring_values(x, y, points, values);
			
			if (jmp == -1)
			{
				(void) (rank == 0) ? free(buffer) : 0;
				free(recv_buffer);
				fprintf(stderr, "[!] Comm error at line %d\n", __LINE__);
				MPI_Finalize();
				return -1;
			}
			
			// Enumerating boundary cases
			
			
			/* If on a boundary, compute the average of the neighboring cells.
			   If on a internal cell, compute the FTCS (Forward in time, Central in space) */
			     
			switch(jmp)
			{
			// 0
			case 1:
				new_points[y*nx + x] = (values[EAST] / 2.0f + values[SOUTH] / 2.0f);
				break;
			// 1
			case 2:
				new_points[y*nx + x] = (values[EAST] / 2.0f + values[NORTH] / 2.0f);
				break;
			// 2
			case 4:
				new_points[y*nx + x] = (values[WEST] / 2.0f + values[SOUTH] / 2.0f);
				break;
			// 3
			case 8:
				new_points[y*nx + x] = (values[WEST] / 2.0f + values[NORTH] / 2.0f);
				break;
			// 4
			case 16:
				new_points[y*nx + x] = values[EAST] / 3.0f + values[NORTH] / 3.0f
					+ values[SOUTH] / 3.0f;
				break;
			// 5
			case 32:
				new_points[y*nx + x] = values[WEST] / 3.0f + values[SOUTH] / 3.0f
					+ values[EAST] / 3.0f;
				break;
			
			// 6
			case 64:
				new_points[y*nx + x] = values[NORTH] / 3.0f + values[WEST] / 3.0f
					+ values[SOUTH] / 3.0f;
				break;
			
			// 7
			case 128:
				new_points[y*nx + x] = values[WEST] / 3.0f + values[NORTH] / 3.0f
					+ values[EAST] / 3.0f;
				break;
				
			// 8	
			case 256:
				
				//get_neighboring_temperatures(stencil);
				new_points[y*nx + x] = points[y*nx + x] + dt * alpha * (
				
					(values[WEST] - (2.0f * points[y*nx + x]) + values[EAST])
							* (1.0f/dx_squared)
					+
					(values[NORTH] - (2.0f * points[y*nx + x]) + values[SOUTH])
							* (1.0f/dy_squared)
				);
				
			default:
				break;
			}
			
		
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
	}
	
	// Last checkpoint
	checkpoint(f, new_points, 0);
	
	fclose(f);
	free(buffer);
	free(recv_buffer);
	
	/*
	// print ending points
	printf("Ending State: \n");
	print_points(points);
	*/
	
	MPI_Finalize();
	return 0;
}
