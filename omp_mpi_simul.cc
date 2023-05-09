// mpic++ -O3 -fno-rtti -fno-exceptions -fopenmp -o d.out omp_mpi_simul.cc

#include <stdio.h>
#include <cmath>
#include <errno.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <numeric>
#include <array>
#include <iomanip>


// Max iters
#define MAX_ITERS 500
#define CHECKPOINT 100

// Length of the domain
#define Lx 90
#define Ly 90

// Discretized differentials
#define dx 1.0f
#define dy 1.0f
#define dt 0.01f


// Thermal diffusivity
#define alpha 0.5

// Graph filename format
#define FILENAME "heat_diffusion_%d.dat"

// Error log name format
#define ERROR_LOG_FORMAT "error_log_%d.txt"

// Maximum size of a path on the system
#ifndef MAX_PATH
#define MAX_PATH 260
#endif

static size_t nx,ny;
static std::ofstream errstream;

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
	REMOTE_VALUE
};


void print_graph(std::ofstream& f, const float* points)
{
	size_t y,x;
	
	for (y=0; y<ny; ++y)
	{
		for (x=0; x<nx; ++x)
		{
			f << std::fixed << std::setprecision(4) << x << " " << y << " " << points[y * nx + x] << "\n";
		}
	}
}

void print_points(std::ofstream& f, const float* points)
{
	size_t y;
	size_t x;
	
	for (y=0; y<ny; ++y)
	{
		for (x=0; x<nx; ++x)
		{
			f << points[y * nx + x] << " ";
		}
		f << "\n";
	}
}

void checkpoint(std::ofstream& f, const float* points, size_t process_rank)
{
	static size_t c = 0;
	
	errstream << "[" << process_rank << "] Checkpoint " << ++c << "\n";
	const float *temp = points - (CHECKPOINT-1) * nx * ny;

	for (size_t i = 0; i < CHECKPOINT; ++i)
	{
		print_graph(f, temp + i * ny * nx);
		f << "\n\n";
	}
}




int main(int argc, char** argv)
{
	// Error variables
	int err;
	char estring[MPI_MAX_ERROR_STRING] = {0};

	// Neighbors array
	int neighbors[4];

	// Initialize MPI
	err = MPI_Init(&argc, &argv);
	if (err)
	{
		MPI_Error_string(err, estring, nullptr);
		std::cerr << "MPI ERROR: " << estring << std::endl;
		MPI_Finalize();
		return err;
	}
	
	// Get the total number of nodes
	int n_nodes;
	err = MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
	
	if (err)
	{
		MPI_Error_string(err, estring, nullptr);
		std::cerr << "MPI ERROR: " << estring << std::endl;
		MPI_Finalize();
		return err;
	}
	
	// Get the rank of this process
	int rank;
	err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if (err)
	{
		MPI_Error_string(err, estring, nullptr);
		std::cerr << "MPI ERROR: " << estring << std::endl;
		MPI_Abort(MPI_COMM_WORLD, err);
		MPI_Finalize();
		return err;
	}

	// Initialize error log stream
	char errstream_filepath[MAX_PATH];
	sprintf(errstream_filepath, ERROR_LOG_FORMAT, rank);
 	errstream.open(errstream_filepath, std::ofstream::out);
	
	if (!errstream.is_open())
	{
		std::cerr << "Could not open error log file\n";
		MPI_Abort(MPI_COMM_WORLD, err);
		MPI_Finalize();
		return -1;
	}

	// Calculate size of chunk
	size_t sqrt_n_nodes_x = (size_t) std::floor(std::sqrt((float)n_nodes));
	size_t sqrt_n_nodes_y = (size_t) std::ceil(std::sqrt((float)n_nodes));
	size_t full_ny, full_nx;

	errstream << "sqrt_n_nodes_x : " << sqrt_n_nodes_x << "\tsqrt_n_nodes_y : " << sqrt_n_nodes_y << std::endl;
	
	// Allocate memory and scatter domain from master to slaves
	std::unique_ptr<float[]> buffer;
	std::unique_ptr<float[]> recv_buffer;
	
	if (rank == 0)
	{
		full_ny = Ly + 1;
		full_nx = Lx + 1;
		
		buffer = std::make_unique<float[]>(full_ny * full_nx * (CHECKPOINT+1));
		
		if (buffer == nullptr)
		{
			errstream << "Error at line " << __LINE__ << std::endl;
			MPI_Abort(MPI_COMM_WORLD, err);
			MPI_Finalize();
			return err;
		}
		
		// Fill for an initial state with center point at 100 °C
		float *tmp = buffer.get();
		for (size_t y=0; y<full_ny; ++y)
		{
			for (size_t x=0; x<full_nx; ++x)
			{
				tmp[y * full_nx + x] = (x == (full_nx / 2) && y == (full_ny / 2)) ? 400.0f : 19.0f;
			}
		}
	
	}
	
	nx = (Lx + 1) / sqrt_n_nodes_x;
	ny = (Ly + 1) / sqrt_n_nodes_y;
	
	size_t rem_nx = (Lx + 1) % sqrt_n_nodes_x;
	size_t rem_ny = (Ly + 1) % sqrt_n_nodes_y;
	
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
				to = (((i/ny) < sqrt_n_nodes_y) ? (i/ny) : (i/ny)-1) * sqrt_n_nodes_x + j;
				count = nx + ((j == sqrt_n_nodes_x-1) ? rem_nx : 0);
				
				errstream << "Sending line " << i << ", chunk " << j << " to " << to << ": " << count << " floats" << std::endl;
				err = MPI_Send(starting_ptr, count, MPI_FLOAT, to, FRAME_INIT, MPI_COMM_WORLD);
				
				if (err)
				{
					MPI_Error_string(err, estring, nullptr);
					errstream << "Error at line " << __LINE__ << "\n";
					errstream << "MPI ERROR: " << estring << std::endl;
					MPI_Abort(MPI_COMM_WORLD, err);
					MPI_Finalize();
					return err;
				}
			}
		}
	}
	
	// Receive initial frame from master node
	nx += (((rank+1) % sqrt_n_nodes_x == 0)) ? rem_nx : 0;
	ny += (rank >= (sqrt_n_nodes_y-1)*sqrt_n_nodes_x) ? rem_ny : 0;

	recv_buffer = std::make_unique<float[]>(nx * ny * (CHECKPOINT+1));

	if (recv_buffer == nullptr)
	{
		errstream << "Error at line " << __LINE__ << std::endl;
		MPI_Abort(MPI_COMM_WORLD, err);
		MPI_Finalize();
		return err;
	}

	errstream << "nx : " << nx << " ny : " << ny << std::endl;
	
	
	for (size_t y=0; y<ny; ++y)
	{
		err = MPI_Recv(recv_buffer.get() + y*nx, nx, MPI_FLOAT, 0, FRAME_INIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		errstream << "Received line " << y << " from 0: " << nx << " floats" << std::endl;

		if (err)
		{
			MPI_Error_string(err, estring, nullptr);
			errstream << "Error at line " << __LINE__ << "\n";
			errstream << "MPI ERROR: " << estring << std::endl;
			MPI_Abort(MPI_COMM_WORLD, err);
			MPI_Finalize();
			return err;
		}
	}
	
	// Initialize spatial differentials
	const float dx_squared = std::pow(dx, 2);
	const float dy_squared = std::pow(dy, 2);
	
	// Get own filename
	char filename[MAX_PATH];
	sprintf(filename, FILENAME, rank);
	
	// Open file to store the results in
	std::ofstream f(filename, std::ofstream::out);

	if (!f.is_open())
	{
		errstream << "Error at line " << __LINE__ << std::endl;
		MPI_Abort(MPI_COMM_WORLD, err);
		MPI_Finalize();
		return -1;
	}
	
	errstream << "Cores: " << omp_get_max_threads() << std::endl;
	
	
	// NORTH
	std::array<std::unique_ptr<float[]>, 4> remote_data;

	if (rank >= sqrt_n_nodes_x)
	{ 
		neighbors[NORTH] = (rank - sqrt_n_nodes_x);
		remote_data[NORTH] = std::make_unique<float[]>(nx);

		//memcpy(remote_data[NORTH].get(), recv_buffer.get(), sizeof(float) * nx);
	}
	else
	{
		neighbors[NORTH] = -1;
	}
	

	// WEST
	if (rank % sqrt_n_nodes_x != 0)
	{
		neighbors[WEST] = (rank - 1);
		remote_data[WEST] = std::make_unique<float[]>(ny);
		
		/*
		for (size_t i=0; i<ny; ++i)
		{
			remote_data[WEST][i] = recv_buffer[i*nx];
		}
		*/
	}
	else
	{
		neighbors[WEST] = -1;
	}
	
	// SOUTH
	if (rank < (sqrt_n_nodes_y-1) * (sqrt_n_nodes_x))
	{
		neighbors[SOUTH] = (rank + sqrt_n_nodes_x);
		remote_data[SOUTH] = std::make_unique<float[]>(nx);

		//memcpy(remote_data[SOUTH].get(), recv_buffer.get() + (ny-1) * nx, sizeof(float) * nx);
	}
	else
	{
		neighbors[SOUTH] = -1;
	}
	
	// EAST
	if ((rank+1) % sqrt_n_nodes_x != 0)
	{
		neighbors[EAST] = (rank + 1);
		remote_data[EAST] = std::make_unique<float[]>(ny);

		/*
		for (size_t i=0; i<ny; ++i)
		{
			remote_data[EAST][i] = recv_buffer[i*nx + (nx-1)];
		}
		*/
	}
	else
	{
		neighbors[EAST] = -1;
	}
	
	// Pointers to frame
	float *points = recv_buffer.get();
	float *new_points = points + ny * nx;

	// print initial points
	errstream << "Initial frame:" << std::endl;
	print_points(errstream, points);
	errstream.flush();

	// Strided mpi vector for column exchange
	MPI_Datatype MPI_Strided_vector;
	MPI_Type_vector(ny, 1, nx, MPI_FLOAT, &MPI_Strided_vector);
	MPI_Type_commit(&MPI_Strided_vector);

	// checkpoint counter
	size_t c = 0;
	
	// timestep variable
	size_t t;

	// Debug: print my neighbors
	errstream << "Neighbors: ";
	for (size_t i=0; i<4; ++i)
	{
		errstream << neighbors[i] << " ";
	}
	errstream << std::endl;

	
	for (t=1; t<MAX_ITERS; ++t)
	{
		
		// Checkpointing procedure
		if (t % CHECKPOINT == 0)
		{
			checkpoint(f, new_points, rank);
		}

		std::array<MPI_Request, 4> requests;

		// Send, receive neighboring data
		for (int n=0; n<4; ++n)
		{
			if (neighbors[n] != -1)
			{
				errstream << "Sending " << n << " border data to " << neighbors[n] << std::endl;

				switch (n)
				{
					case NORTH:
					err = MPI_Send(recv_buffer.get(), nx, MPI_FLOAT, neighbors[n], REMOTE_VALUE, MPI_COMM_WORLD);
					break;

					case WEST:
					err = MPI_Send(recv_buffer.get(), 1, MPI_Strided_vector, neighbors[n], REMOTE_VALUE, MPI_COMM_WORLD);
					break;

					case SOUTH:
					err = MPI_Send(recv_buffer.get()+((ny-1)*nx), nx, MPI_FLOAT, neighbors[n], REMOTE_VALUE, MPI_COMM_WORLD);
					break;

					case EAST:
					err = MPI_Send(recv_buffer.get()+(nx-1), 1, MPI_Strided_vector, neighbors[n], REMOTE_VALUE, MPI_COMM_WORLD);
					break;

					default:
					break;
				}

				if (err)
				{
					MPI_Error_string(err, estring, nullptr);
					errstream << "Error at line " << __LINE__ << "\n";
					errstream << "MPI ERROR: " << estring << std::endl;
					MPI_Abort(MPI_COMM_WORLD, err);
					MPI_Finalize();
					return err;
				}

				errstream << "Deferred Receive " << n << " border data from " << neighbors[n] << std::endl;
				err = MPI_Irecv(remote_data[n].get(), (n % 2) ? ny : nx, MPI_FLOAT, neighbors[n], REMOTE_VALUE, MPI_COMM_WORLD, &(requests[n]));

				if (err)
				{
					MPI_Error_string(err, estring, nullptr);
					errstream << "Error at line " << __LINE__ << "\n";
					errstream << "MPI ERROR: " << estring << std::endl;
					MPI_Abort(MPI_COMM_WORLD, err);
					MPI_Finalize();
					return err;
				}
			}
		}
				
		// Adjust pointers to next frame
		points = recv_buffer.get() + ((t-1) % CHECKPOINT) * ny * nx;
		new_points = recv_buffer.get() + (t % CHECKPOINT) * ny * nx;
	
		
		// Grid points computation
		#pragma omp parallel for shared(err)
		for (size_t i=nx; i<((ny-1)*nx); ++i)
		{

			size_t y = i / nx;
			size_t x = i % nx;

			if (x == 0 || x == nx-1)
				continue;
			
			float stencil[4] = {points[(y-1)*nx + x],
				points[y*nx + x - 1],
				points[(y+1)*nx + x],
				points[y*nx + x + 1]};
				
			new_points[y*nx + x] = points[y*nx + x] + dt * alpha * (
			
				(stencil[WEST] - 2.0f*points[y*nx + x] + stencil[EAST])
						* (1.0f/dx_squared)
				+
				(stencil[NORTH] - 2.0f*points[y*nx + x] + stencil[SOUTH])
						* (1.0f/dy_squared)
			);
		
		}
		
		if (err)
		{
			errstream << "Error at line " << __LINE__ << std::endl;
			MPI_Abort(MPI_COMM_WORLD, err);
			MPI_Finalize();
			return -1;
		}
		
		// Wait for border data
		for (int n=0; n<4; ++n)
		{
			if (neighbors[n] != -1)
			{
				errstream << "MPI_Wait on border data..." << std::endl;
				MPI_Wait(&(requests[n]), MPI_STATUS_IGNORE);
				errstream << "MPI_Wait unlocked!" << std::endl;
			}
		}

		// Deal with boundaries, excluding corners
		#pragma omp parallel for
		for (int n=0; n<4; ++n)
		{
			std::array<float, 4> stencil = {0.0f};

			if (neighbors[n] != -1)
			{

				for (size_t i=1; i<((n % 2) ? (ny-1) : (nx-1)); ++i)
				{
					
					size_t k;

					switch (n)
					{
					case NORTH:
						stencil[NORTH] = remote_data[NORTH][i];
						stencil[SOUTH] = points[nx + i];
						stencil[WEST] = points[i-1];
						stencil[EAST] = points[i+1];
						k = i;
						break;

					case SOUTH:
						stencil[NORTH] = points[(ny-2)*nx + i];
						stencil[SOUTH] = remote_data[SOUTH][i];
						stencil[WEST] = points[(ny-1)*nx + i - 1];
						stencil[EAST] = points[(ny-1)*nx + i + 1];
						k = (ny-1)*nx + i;
						break;

					case WEST:
						stencil[NORTH] = points[(i-1)*nx];
						stencil[SOUTH] = points[(i+1)*nx];
						stencil[WEST] = remote_data[WEST][i];
						stencil[EAST] = points[i*nx + 1];
						k = i*nx;
						break;
					
					case EAST:
						stencil[NORTH] = points[(i-1)*nx + nx-1];
						stencil[SOUTH] = points[(i+1)*nx + nx-1];
						stencil[WEST] = points[i*nx + (nx-2)];
						stencil[EAST] = remote_data[EAST][i];
						k = i*nx + nx-1;
						break;

					default:
						break;
					}

					new_points[k] = points[k] + dt * alpha * (
			
					(stencil[WEST] - 2.0f*points[k] + stencil[EAST])
							* (1.0f/dx_squared)
					+
					(stencil[NORTH] - 2.0f*points[k] + stencil[SOUTH])
							* (1.0f/dy_squared)
					);
				}
			}
			else
			{
				for (size_t i=1; i<((n % 2) ? (ny-1) : (nx-1)); ++i)
				{
					
					size_t k;

					switch (n)
					{
					case NORTH:
						stencil[SOUTH] = points[nx + i];
						stencil[WEST] = points[i-1];
						stencil[EAST] = points[i+1];
						k = i;
						break;

					case SOUTH:
						stencil[NORTH] = points[(ny-2)*nx + i];
						stencil[WEST] = points[(ny-1)*nx + i - 1];
						stencil[EAST] = points[(ny-1)*nx + i + 1];
						k = (ny-1)*nx + i;
						break;

					case WEST:
						stencil[NORTH] = points[(i-1)*nx];
						stencil[SOUTH] = points[(i+1)*nx];
						stencil[EAST] = points[i*nx + 1];
						k = i*nx;
						break;
					
					case EAST:
						stencil[NORTH] = points[(i-1)*nx + nx-1];
						stencil[SOUTH] = points[(i+1)*nx + nx-1];
						stencil[WEST] = points[i*nx + (nx-2)];
						k = i*nx + nx-1;
						break;

					default:
						break;
					}

					new_points[k] = std::accumulate(std::begin(stencil), std::end(stencil), 0.0f) / 3.0f;
				}
			}
		}

		// Compute corners

		// TOP LEFT
		{
			std::array<float, 4> stencil = {0.0f};
			int f = 2;

			stencil[SOUTH] = points[nx];
			stencil[EAST] = points[1];
			stencil[NORTH] = ((neighbors[NORTH] != -1)) ? (--f, remote_data[NORTH][0]) : 0.0f;
			stencil[WEST] = ((neighbors[WEST] != -1)) ? (--f, remote_data[WEST][0]) : 0.0f;

			if (f > 0)
			{
				if (f == 2)
				{
						new_points[0] = std::accumulate(std::begin(stencil), std::end(stencil), 0.0f)
							/ 2.0f;
				}
				else
				{
				
				new_points[0] = std::accumulate(std::begin(stencil), std::end(stencil), 0.0f)
						/ 3.0f;
				}
				
			}
			else
			{
				new_points[0] = points[0] + dt * alpha * (

					(stencil[WEST] - 2.0f*(points[0]) + stencil[EAST])
							* (1.0f/dx_squared)
					+
					(stencil[NORTH] - 2.0f*(points[0]) + stencil[SOUTH])
							* (1.0f/dy_squared)
					);
			}
		}

		// TOP RIGHT
		{
			std::array<float, 4> stencil = {0.0f};
			int f = 2;

			stencil[SOUTH] = points[nx + nx-1];
			stencil[WEST] = points[nx-2];
			stencil[NORTH] = ((neighbors[NORTH] != -1)) ? (--f, remote_data[NORTH][nx-1]) : 0.0f;
			stencil[EAST] = ((neighbors[EAST] != -1)) ? (--f, remote_data[EAST][0]) : 0.0f;

			if (f > 0)
			{
				if (f == 2)
				{
						new_points[nx-1] = std::accumulate(std::begin(stencil), std::end(stencil), 0.0f)
							/ 2.0f;
				}
				else
				{
				
					new_points[nx-1] = std::accumulate(std::begin(stencil), std::end(stencil), 0.0f)
						/ 3.0f;
				}
				
			}
			else
			{
				new_points[nx-1] = points[nx-1] + dt * alpha * (

					(stencil[WEST] - 2.0f*(points[nx-1]) + stencil[EAST])
							* (1.0f/dx_squared)
					+
					(stencil[NORTH] - 2.0f*(points[nx-1]) + stencil[SOUTH])
							* (1.0f/dy_squared)
					);
			}
		}

		// BOTTOM LEFT
		{
			std::array<float, 4> stencil = {0.0f};
			int f = 2;

			stencil[NORTH] = points[(ny-2)*nx];
			stencil[EAST] = points[(ny-1)*nx + 1];
			stencil[SOUTH] = ((neighbors[SOUTH] != -1)) ? (--f, remote_data[SOUTH][0]) : 0.0f;
			stencil[WEST] = ((neighbors[WEST] != -1)) ? (--f, remote_data[WEST][ny-1]) : 0.0f;

			if (f > 0)
			{
				if (f == 2)
				{
						new_points[(ny-1)*nx] = std::accumulate(std::begin(stencil), std::end(stencil), 0.0f)
							/ 2.0f;
				}
				else
				{
				
				new_points[(ny-1)*nx] = std::accumulate(std::begin(stencil), std::end(stencil), 0.0f)
						/ 3.0f;
				}
				
			}
			else
			{
				new_points[(ny-1)*nx] = points[(ny-1)*nx] + dt * alpha * (

					(stencil[WEST] - 2.0*(points[(ny-1)*nx]) + stencil[EAST])
							* (1.0f/dx_squared)
					+
					(stencil[NORTH] - 2.0*(points[(ny-1)*nx]) + stencil[SOUTH])
							* (1.0f/dy_squared)
					);
			}
		}

		// BOTTOM RIGHT
		{
			std::array<float, 4> stencil = {0.0f};
			int f = 2;

			stencil[NORTH] = points[(ny-2)*nx + nx - 1];
			stencil[WEST] = points[(ny-1)*nx + nx - 2];
			stencil[SOUTH] = ((neighbors[SOUTH] != -1)) ? (--f, remote_data[SOUTH][nx-1]) : 0.0f;
			stencil[EAST] = ((neighbors[EAST] != -1)) ? (--f, remote_data[EAST][ny-1]) : 0.0f;

			if (f > 0)
			{
				if (f == 2)
				{
						new_points[(ny-1)*nx+nx-1] = std::accumulate(std::begin(stencil), std::end(stencil), 0.0f)
							/ 2.0f;
				}
				else
				{
				
				new_points[(ny-1)*nx+nx-1] = std::accumulate(std::begin(stencil), std::end(stencil), 0.0f)
						/ 3.0f;
				}
				
			}
			else
			{
				new_points[(ny-1)*nx+nx-1] = points[(ny-1)*nx+nx-1] + dt * alpha * (

					(stencil[WEST] - 2.0f*(points[(ny-1)*nx+nx-1]) + stencil[EAST])
							* (1.0f/dx_squared)
					+
					(stencil[NORTH] - 2.0f*(points[(ny-1)*nx+nx-1]) + stencil[SOUTH])
							* (1.0f/dy_squared)
					);
			}
		}

		// Synchronization point
		errstream << "***BARRIER***" << std::endl;
		MPI_Barrier(MPI_COMM_WORLD);
	}
	
	// Last checkpoint
	checkpoint(f, new_points, rank);
	
	// Send last frame chunk to master
	for (size_t y=0; y<ny; ++y)
	{
		err = MPI_Send(recv_buffer.get() + y*nx, nx, MPI_FLOAT, 0, FRAME_END, MPI_COMM_WORLD);

		if (err)
		{
			MPI_Error_string(err, estring, nullptr);
			errstream << "Error at line " << __LINE__ << "\n";
			errstream << "MPI ERROR: " << estring << std::endl;
			MPI_Abort(MPI_COMM_WORLD, err);
			MPI_Finalize();
			return err;
		}
	}
	
	// Collect last frame chunks from slaves
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
				from = (((i/ny) < sqrt_n_nodes_y) ? (i/ny) : (i/ny)-1) * sqrt_n_nodes_x + j;
				count = nx + ((j == sqrt_n_nodes_x-1) ? rem_nx : 0);
				
				err = MPI_Recv(starting_ptr, count, MPI_FLOAT, from, FRAME_END, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				
				if (err)
				{
					MPI_Error_string(err, estring, nullptr);
					errstream << "Error at line " << __LINE__ << "\n";
					errstream << "MPI ERROR: " << estring << std::endl;
					MPI_Abort(MPI_COMM_WORLD, err);
					MPI_Finalize();
					return err;
				}
			}
		}
	}

	// print ending points
	errstream << "Ending Frame:" << std::endl;
	print_points(errstream, points);
	errstream.flush();
	
	MPI_Finalize();
	return 0;
}
