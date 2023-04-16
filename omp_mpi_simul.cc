#include <stdio.h>
#include <cmath>
#include <errno.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <thread>
#include <string>
#include <fstream>

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
#define FILENAME "heat_diffusion_%d.dat"

#ifndef MAX_PATH
#define MAX_PATH 260
#endif

static size_t nx,ny;
static int neighbors[4];
static char estring[MPI_MAX_ERROR_STRING] = {0};
static float *current_frame;

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
	FRAME_END,
	REQ_VALUE,
	SUPPL_VALUE
};

void respond_routine(bool *terminate)
{
	MPI_Request req[4];
	size_t request_arr[2*4];

	for (size_t i=0; i<4; ++i)
	{
		if (neighbors[i] == -1)
		{
			continue;
		}

		MPI_Irecv(request_arr+i*2, 2, MPI_UNSIGNED_LONG, neighbors[i], REQ_VALUE, MPI_COMM_WORLD, &req[i]);
	}

	while (!(*terminate))
	{
		// For every neighbor, check if it has requested values
		for (size_t i=0; i<4; ++i)
		{
			// No neighbor = skip
			if (neighbors[i] == -1)
			{
				continue;
			}

			int flag;
			MPI_Test(req+i, &flag, MPI_STATUS_IGNORE);

			if (flag)
			{
				size_t x = request_arr[i*3];
				size_t y = request_arr[i*3+1];

				x = std::min(x, nx-1);
				y = std::min(y, ny-1);

				float val = current_frame[y*nx + x];

				// Respond with requested values
				MPI_Send(&val, 1, MPI_FLOAT, neighbors[i], SUPPL_VALUE, MPI_COMM_WORLD);

				// Listen for further requests
				MPI_Irecv(request_arr+i*2, 2, MPI_UNSIGNED_LONG, neighbors[i], REQ_VALUE, MPI_COMM_WORLD, &req[i]);
			}
		}
	}
}

void print_graph(std::ofstream& f, const float* points)
{
	size_t y,x;
	
	for (y=0; y<ny; ++y)
	{
		for (x=0; x<nx; ++x)
		{
			f << x << " " << y << " " << points[y * nx + x] << "\n";
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

void checkpoint(std::ofstream& f, const float* points, size_t process_rank)
{
	static size_t c = 0;
	
	printf("[%lu] Checkpoint %lu\n", process_rank, ++c);
	const float* temp = points - (CHECKPOINT-1) * nx * ny;

	for (size_t i = 0; i < CHECKPOINT; ++i)
	{
		print_graph(f, temp + i * ny * nx);
		f << "\n\n";
	}
}

template <size_t DIRECTION>
int get_remote_value(size_t x, float* out)
{
	size_t arr[3] = {0, nx, ny};
	int err;
	
	
	if constexpr (DIRECTION == NORTH)
	{
		// SOUTH of neighbor
		arr[0] = (ny-1)*nx + x;
		
		#pragma omp critical(critical_north)
		{
			err = MPI_Send(arr, 3, MPI_UNSIGNED_LONG, neighbors[NORTH], REQ_VALUE, MPI_COMM_WORLD);
			
			err = MPI_Recv(out, 1, MPI_FLOAT, neighbors[NORTH], SUPPL_VALUE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		}
		
		if (err)
		{
			MPI_Error_string(err, estring, nullptr);
			fprintf(stderr, "MPI ERROR: %s\n", estring);
			//MPI_Finalize();
		}

	}
	
	if constexpr (DIRECTION == WEST)
	{
		// EAST of neighbor
		arr[0] = x * nx + nx-1;
		
		#pragma omp critical(critical_west)
		{
			err = MPI_Send(arr, 3, MPI_UNSIGNED_LONG, neighbors[WEST], REQ_VALUE, MPI_COMM_WORLD);
			
			err = MPI_Recv(out, 1, MPI_FLOAT, neighbors[WEST], SUPPL_VALUE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		
		if (err)
		{
			MPI_Error_string(err, estring, nullptr);
			fprintf(stderr, "MPI ERROR: %s\n", estring);

		}
	
	}
	
	if constexpr (DIRECTION == SOUTH)
	{
		// NORTH of neighbor
		arr[0] = x;
		
		
		#pragma omp critical(critical_south)
		{
			err = MPI_Send(arr, 3, MPI_UNSIGNED_LONG, neighbors[SOUTH], REQ_VALUE, MPI_COMM_WORLD);
			
			err = MPI_Recv(out, 1, MPI_FLOAT, neighbors[SOUTH], SUPPL_VALUE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		
		if (err)
		{
			MPI_Error_string(err, estring, nullptr);
			fprintf(stderr, "MPI ERROR: %s\n", estring);
		}
	}
	
	if constexpr (DIRECTION == EAST)
	{
		// WEST of neighbor
		arr[0] = x*nx;
		
		
		#pragma omp critical(critical_east)
		{
			err = MPI_Send(arr, 3, MPI_UNSIGNED_LONG, neighbors[EAST], REQ_VALUE, MPI_COMM_WORLD);
			
			err = MPI_Recv(out, 1, MPI_FLOAT, neighbors[EAST], SUPPL_VALUE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		
		
		if (err)
		{
			MPI_Error_string(err, estring, nullptr);
			fprintf(stderr, "MPI ERROR: %s\n", estring);
		}
	}
	
	return err;
}

float fdivide2(const float f)
{
	int exp;
	int mantix = std::frexp(f, &exp);
	
	return std::ldexp(mantix, --exp);
}

float fmul2(const float f)
{
	int exp;
	int mantix = std::frexp(f, &exp);
	
	return std::ldexp(mantix, ++exp);
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
			
		}
		else
		{
			out_values[NORTH] = fdivide2(out_values[EAST] + out_values[SOUTH]);
		}
		
		if (neighbors[WEST] != -1)
		{
			err = get_remote_value<WEST>(y, &out_values[WEST]);
			if (err)
			{
				return -1;
			}
			
		}
		else
		{
			out_values[WEST] = fdivide2(out_values[EAST] + out_values[SOUTH]);
		}
		
		return 0;
	}
	
	if (x == 0 && y == (ny-1))
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
		}
		else
		{
			out_values[SOUTH] = fdivide2(out_values[EAST] + out_values[NORTH]);
		}
	
		if (neighbors[WEST] != -1)
		{
			err = get_remote_value<WEST>(y, &out_values[WEST]);
			if (err)
			{
				return -1;
			}
			
		}
		else
		{
			out_values[WEST] = fdivide2(out_values[EAST] + out_values[NORTH]);
		}
		
		return 0;
	}
	
	if (x == (nx-1) && y == 0)
	{
		out_values[WEST] = points[y*nx + x - 1];
		out_values[SOUTH] = points[(y+1)*nx + x];
		
		if (neighbors[NORTH] != -1)
		{
			err = get_remote_value<NORTH>(x, &out_values[NORTH]);
			
			if (err)
			{
				return -1;
			}
			
		}
		else
		{
			out_values[NORTH] = fdivide2(out_values[WEST] + out_values[SOUTH]);
		}
		
		if (neighbors[EAST] != -1)
		{
			err = get_remote_value<EAST>(y, &out_values[EAST]);
			
			if (err)
			{
				return -1;
			}
		}
		else
		{
			out_values[EAST] = fdivide2(out_values[WEST] + out_values[SOUTH]);
		}
	
		return 0;
	}
	
	if (x == (nx-1) && y == (ny-1))
	{
		out_values[WEST] = points[y*nx + x - 1];
		out_values[NORTH] = points[(y+1)*nx + x];
		
		if (neighbors[SOUTH] != -1)
		{
			err = get_remote_value<SOUTH>(x, &out_values[SOUTH]);
			
			if (err)
			{
				return -1;
			}
		}
		else
		{
			out_values[SOUTH] = fdivide2(out_values[WEST] + out_values[NORTH]);
		}
		
		if (neighbors[EAST] != -1)
		{
			err = get_remote_value<EAST>(y, &out_values[EAST]);
			
			if (err)
			{
				return -1;
			}	
		}
		else
		{
			out_values[EAST] = fdivide2(out_values[WEST] + out_values[NORTH]);
		}
		
		return 0;
	}
	
	
	if (x == 0 && y != 0 && y != (ny-1))
	{
		out_values[NORTH] = points[(y-1)*nx + x];
		out_values[SOUTH] = points[(y+1)*nx + x];
		out_values[EAST] = points[y*nx + x + 1];
		
		if (neighbors[WEST] != -1)
		{
			err = get_remote_value<WEST>(y, &out_values[WEST]);
			
			if (err)
			{
				return -1;
			}
		}
		else
		{
			out_values[WEST] = (out_values[NORTH] + out_values[SOUTH] + out_values[EAST]) / 3.0f;
		}
		
		return 0;
	}
	
	
	if (y == 0 && x != 0 && x != (nx-1))
	{
		out_values[WEST] = points[y*nx + x - 1];
		out_values[SOUTH] = points[(y+1)*nx + x];
		out_values[EAST] = points[y*nx + x + 1];
		
		if (neighbors[NORTH] != -1)
		{
			err = get_remote_value<NORTH>(y, &out_values[NORTH]);
			
			if (err)
			{
				return -1;
			}
		}
		else
		{
			out_values[NORTH] = (out_values[WEST] + out_values[SOUTH] + out_values[EAST]) / 3.0f;
		}
		
		return 0;
	}
	
	if (x == (nx-1) && y != 0 && y != (ny-1))
	{
		out_values[WEST] = points[y*nx + x - 1];
		out_values[SOUTH] = points[(y+1)*nx + x];
		out_values[NORTH] = points[(y-1)*nx + x];
		
		if (neighbors[EAST] != -1)
		{
			err = get_remote_value<EAST>(y, &out_values[EAST]);
			
			if (err)
			{
				return -1;
			}
		}
		else
		{
			out_values[EAST] = (out_values[WEST] + out_values[SOUTH] + out_values[NORTH]) / 3.0f;
		}
		
		return 0;
	}
	
	
	if (y == (ny-1) && x != 0 && x != (nx-1))
	{
		out_values[WEST] = points[y*nx + x - 1];
		out_values[EAST] = points[y*nx + x + 1];
		out_values[NORTH] = points[(y-1)*nx + x];
		
		if (neighbors[SOUTH] != -1)
		{
			err = get_remote_value<SOUTH>(y, &out_values[SOUTH]);
			
			if (err)
			{
				return -1;
			}
		}
		else
		{
			out_values[SOUTH] = (out_values[WEST] + out_values[EAST] + out_values[NORTH]) / 3.0f;
		}
		
		return 0;
	}
	
	out_values[NORTH] = points[(y-1)*nx + x];
	out_values[WEST] = points[y*nx + x - 1];
	out_values[EAST] = points[y*nx + x + 1];
	out_values[SOUTH] = points[(y+1)*nx + x];
	
	return 0;
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
	err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
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
	std::unique_ptr<float> buffer;
	std::unique_ptr<float> recv_buffer;
	
	if (rank == 0)
	{
		full_ny = Ly + 1;
		full_nx = Lx + 1;
		
		buffer = std::make_unique<float>(full_ny * full_nx * (CHECKPOINT+1));
		
		if (buffer == nullptr)
		{
			MPI_Error_string(err, estring, nullptr);
			fprintf(stderr, "MPI ERROR: %s\n", estring);
			MPI_Abort(MPI_COMM_WORLD, err);
			MPI_Finalize();
			return err;
		}
		
		// Fill for an initial state with center point at 100 °C 
		for (size_t y=0; y<full_ny; ++y)
		{
			for (size_t x=0; x<full_nx; ++x)
			{
				buffer.get()[y * full_nx + x] = (x == (full_nx / 2) && y == (full_ny / 2)) ? 100.0f : 19.0f;
			}
		}
	
	}
	
	nx = (Lx + 1) / sqrt_n_nodes_x;
	ny = (Ly + 1) / sqrt_n_nodes_y;
	
	size_t rem_nx = (Lx + 1) % sqrt_n_nodes_x;
	size_t rem_ny = (Ly + 1) % sqrt_n_nodes_y;
	
	
	recv_buffer = std::make_unique<float>((nx+rem_nx) * (ny+rem_ny) * (CHECKPOINT+1));
	
	if (recv_buffer == nullptr)
	{
		MPI_Error_string(err, estring, nullptr);
		fprintf(stderr, "MPI ERROR: %s\n", estring);
		MPI_Abort(MPI_COMM_WORLD, err);
		MPI_Finalize();
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
				starting_ptr = buffer.get() + i*full_nx + j*nx;
				to = (i/ny)*sqrt_n_nodes_x + j;
				count = nx + ((j == sqrt_n_nodes_x-1) ? rem_nx : 0);
				
				err = MPI_Send(starting_ptr, count, MPI_FLOAT, to, FRAME_INIT, MPI_COMM_WORLD);
				
				if (err)
				{
					MPI_Error_string(err, estring, nullptr);
					fprintf(stderr, "MPI ERROR: %s\n", estring);
					MPI_Abort(MPI_COMM_WORLD, err);
					MPI_Finalize();
					return err;
				}
			}
		}
	}
	
	// Receive initial frame from master node
	nx += ((rank+1) % sqrt_n_nodes_x == 0) ? rem_nx : 0;
	ny += (rank+1 - (sqrt_n_nodes_y-1) * sqrt_n_nodes_x >= 0) ? rem_ny : 0;
	
	size_t recv_size = nx * ny;
	
	err = MPI_Recv(recv_buffer.get(), recv_size, MPI_FLOAT, 0, FRAME_INIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	if (err)
	{
		MPI_Error_string(err, estring, nullptr);
		fprintf(stderr, "MPI ERROR: %s\n", estring);
		MPI_Abort(MPI_COMM_WORLD, err);
		MPI_Finalize();
		return err;
	}

	
	// Initialize spatial differentials
	const float dx_squared = std::pow(dx, 2);
	const float dy_squared = std::pow(dy, 2);
	
	// Gridpoint variables
	size_t x, y;
	
	// Get own filename
	char filename[MAX_PATH];
	sprintf(filename, FILENAME, rank);
	
	// Open file to store the results in
	std::ofstream f(filename, std::ofstream::out);

	if (!f.is_open())
	{
		printf("Error at line %d", __LINE__);
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

	
	// Pointers to frame
	float *points = recv_buffer.get();
	float *new_points = points + ny * nx;

	// print initial points
	printf("Initial State: \n");
	print_points(points);


	// Delegate a thread to responding to other nodes
	bool terminate = false;
	std::thread sendback(respond_routine, &terminate);

	// checkpoint counter
	size_t c = 0;
	
	// timestep variable
	size_t t;

	for (t = 1; t<MAX_ITERS; ++t)
	{
		
		// Checkpointing procedure
		if (t % CHECKPOINT == 0)
		{
			checkpoint(f, new_points, rank);
		}
				
		// Adjust pointers to next frame
		points = recv_buffer.get() + ((t-1) % CHECKPOINT) * ny * nx;
		new_points = recv_buffer.get() + (t % CHECKPOINT) * ny * nx;
		current_frame = points;
		
		int err = 0;
		
		// Grid points computation
		#pragma omp parallel for shared(err)
		for (size_t i=0; i<(ny*nx); ++i)
		{
					
			y = i / nx;
			x = i % nx;
			
			float stencil[4];
			
			err = get_neighboring_values(x, y, points, stencil);
				
			new_points[y*nx + x] = points[y*nx + x] + dt * alpha * (
			
				(stencil[WEST] - fmul2(points[y*nx + x]) + stencil[EAST])
						* (1.0f/dx_squared)
				+
				(stencil[NORTH] - fmul2(points[y*nx + x]) + stencil[SOUTH])
						* (1.0f/dy_squared)
			);
		
		}
		
		if (err)
		{
			return -1;
		}
		
		// Synchronization point
		MPI_Barrier(MPI_COMM_WORLD);
	}
	
	// Last checkpoint
	checkpoint(f, new_points, rank);
	
	err = MPI_Send(recv_buffer.get(), recv_size, MPI_FLOAT, 0, FRAME_END, MPI_COMM_WORLD);
	
	if (err)
	{
		MPI_Error_string(err, estring, nullptr);
		fprintf(stderr, "MPI ERROR: %s\n", estring);
		MPI_Abort(MPI_COMM_WORLD, err);
		MPI_Finalize();
		return err;
	}
	
	if (rank == 0)
	{
		for (size_t i=0; i<full_ny; ++i)
		{
			float *starting_ptr;
			int from;
			size_t count;
			
			for (size_t j=0; j<sqrt_n_nodes_x; ++j)
			{
				starting_ptr = buffer.get() + i*full_nx + j*nx;
				from = (i/ny) * sqrt_n_nodes_x + j;
				count = nx + ((j == sqrt_n_nodes_x-1) ? rem_nx : 0);
				
				err = MPI_Recv(starting_ptr, count, MPI_FLOAT, from, FRAME_END, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				
				if (err)
				{
					MPI_Error_string(err, estring, nullptr);
					fprintf(stderr, "MPI ERROR: %s\n", estring);
					MPI_Abort(MPI_COMM_WORLD, err);
					MPI_Finalize();
					return err;
				}
			}
		}
	}
	
	terminate = true;
	sendback.join();

	// print ending points
	printf("Ending State: \n");
	print_points(points);
	
	
	MPI_Finalize();
	return 0;
}
