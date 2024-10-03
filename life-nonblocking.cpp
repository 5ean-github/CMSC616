#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <mpi.h>

using namespace std;

/*
 * Reads the input file line by line and stores it in a 2D matrix.
 */
void read_input_file(int **life, string const &input_file_name) {
    
    // Open the input file for reading.
    ifstream input_file;
    input_file.open(input_file_name);
    if (!input_file.is_open())
        perror("Input file cannot be opened");

    string line, val;
    int x, y;
    while (getline(input_file, line)) {
        stringstream ss(line);
        
        // Read x coordinate.
        getline(ss, val, ',');
        x = stoi(val);
        
        // Read y coordinate.
        getline(ss, val);
        y = stoi(val);

        // Populate the life matrix.
        life[x][y] = 1;
    }
    input_file.close();
}

/* 
 * Writes out the final state of the 2D matrix to a csv file. 
 */
void write_output(int **result_matrix, int X_limit, int Y_limit,
                  string const &input_name, int num_of_generations, int size) {
    
    // Open the output file for writing.
    ofstream output_file;
    string input_file_name = input_name.substr(0, input_name.length() - 5);
    output_file.open(input_file_name + "." + to_string(num_of_generations) + "." + to_string(size) + 
                    "_parallel.csv");
    if (!output_file.is_open())
        perror("Output file cannot be opened");
    
    // Output each live cell on a new line. 
    for (int i = 0; i < X_limit; i++) {
        for (int j = 0; j < Y_limit; j++) {
            if (result_matrix[i][j] == 1) {
                output_file << i << "," << j << "\n";
            }
        }
    }
    output_file.close();
}

/**
  * The main function to execute "Game of Life" simulations on a 2D board.
  */
int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0) {
            std::cerr << "Usage: mpirun -np <# of processes> ./life <data-file-name> <# of generations> <X_limit> <Y_limit>\n";
        }
        MPI_Finalize();
        return -1;
    }

    string input_file_name = argv[1];
    int num_of_generations = stoi(argv[2]);
    int X_limit = stoi(argv[3]);
    int Y_limit = stoi(argv[4]);

    int local_X_limit = X_limit / size;

    int** local_grid = new int*[local_X_limit];
    for (int i = 0; i < local_X_limit; ++i) {
        local_grid[i] = new int[Y_limit]();
    }

    //previous life for local grid 
    int **previous_life = new int *[local_X_limit];
    for (int i = 0; i < local_X_limit; i++) {
        previous_life[i] = new int[Y_limit+2];
        for (int j = 0; j < Y_limit+2; j++) {
            previous_life[i][j] = 0;
        }
    }

    // Master (rank 0) allocates space for the entire global grid
    int** global_grid = nullptr;
    if (rank == 0) {
        global_grid = new int*[X_limit];
        for (int i = 0; i < X_limit; ++i) {
            global_grid[i] = new int[Y_limit]();
        }
        read_input_file(global_grid, input_file_name);
    }

    for (int i = 0; i < local_X_limit; i++) {
        MPI_Scatter(global_grid ? global_grid[i] : MPI_IN_PLACE, Y_limit, MPI_INT, local_grid[i], Y_limit, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Request reqs[4]; // Two sends and two receives
    int* top_ghost_row = new int[Y_limit];
    int* bottom_ghost_row = new int[Y_limit];


    double start_time = MPI_Wtime();

    for (int numg = 0; numg < num_of_generations; numg++) {
        int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;              // Up neighbor (if at top, use MPI_PROC_NULL)
        int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;     // Down neighbor (if at bottom, use MPI_PROC_NULL)

        // Post non-blocking receives for top and bottom ghost rows
        MPI_Irecv(top_ghost_row, Y_limit, MPI_INT, up, 0, MPI_COMM_WORLD, &reqs[0]);  // Receive from above
        MPI_Irecv(bottom_ghost_row, Y_limit, MPI_INT, down, 0, MPI_COMM_WORLD, &reqs[1]);  // Receive from below

        // Post non-blocking sends for sending top and bottom rows
        MPI_Isend(local_grid[0], Y_limit, MPI_INT, up, 0, MPI_COMM_WORLD, &reqs[2]);    // Send top row
        MPI_Isend(local_grid[local_X_limit - 1], Y_limit, MPI_INT, down, 0, MPI_COMM_WORLD, &reqs[3]);  // Send bottom row

        //compute local_grid[1:local_X_limit-1]

        for (int i = 0; i < local_X_limit; i++) {
            for (int j = 0; j < Y_limit; j++) {
                previous_life[i][j + 1] = local_grid[i][j];
            }
        }

        
        //previous_life: local_X_limit*(Y_limit+2)
        int neighbors = 0;
        for (int i = 1; i < local_X_limit - 1; i++) {
            for (int j = 0; j < Y_limit; j++) {
                neighbors = previous_life[i - 1][j] + previous_life[i - 1][j+1] +
                previous_life[i - 1][j + 2] + previous_life[i][j] +
                previous_life[i][j + 2] + previous_life[i + 1][j] +
                previous_life[i + 1][j+1] + previous_life[i + 1][j + 2];

                if (previous_life[i][j+1]==0){
                    if (neighbors == 3)
                    local_grid[i][j] = 1;
                }else {
                // An occupied cell survives only if it has either 2 or 3 neighbors.
                // The cell dies out of loneliness if its neighbor count is 0 or 1.
                // The cell also dies of overpopulation if its neighbor count is 4-8.
                if (neighbors != 2 && neighbors != 3) {
                    local_grid[i][j] = 0;
                }
                }
            }
        }
        

        // Wait for all non-blocking communications to complete
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // After receiving the ghost rows, update the boundary rows
        //compute local_grid[0]
        for (int j=0;j<Y_limit;j++){
            int tgrl = (j==0) ? 0 : top_ghost_row[j-1];
            int tgrr = (j==Y_limit-1) ? 0 : top_ghost_row[j+1];
            neighbors = tgrl + top_ghost_row[j] + tgrr + 
            previous_life[0][j] + previous_life[0][j+2] +
            previous_life[1][j] + previous_life[1][j+1] + previous_life[1][j+2];
            if (previous_life[0][j+1]==0){
                if (neighbors == 3)
                local_grid[0][j] = 1;
            }else {
            if (neighbors != 2 && neighbors != 3) {
                local_grid[0][j] = 0;
            }
            }
        }
        //compute local_grid[local_X_limit-1]
        for (int j=0;j<Y_limit;j++){
            int bgrl = (j==0) ? 0 : bottom_ghost_row[j-1];
            int bgrr = (j==Y_limit-1) ? 0 : bottom_ghost_row[j+1];
            neighbors = bgrl + bottom_ghost_row[j] + bgrr + 
            previous_life[local_X_limit-1][j] + previous_life[local_X_limit-1][j+2] +
            previous_life[local_X_limit-2][j] + previous_life[local_X_limit-2][j+1] + previous_life[local_X_limit-2][j+2];
            if (previous_life[local_X_limit-1][j+1]==0){
                if (neighbors == 3)
                local_grid[local_X_limit-1][j] = 1;
            }else {
            if (neighbors != 2 && neighbors != 3) {
                local_grid[local_X_limit-1][j] = 0;
            }
            }
        }
        if (rank==1 && numg == 18){
            printf("%d\n",numg);
            printf("%d ",local_grid[8][17]);
            printf("%d ",local_grid[8][18]);
            fflush(stdout);
        }
        
    }
    double end_time = MPI_Wtime();
    double runtime = end_time - start_time;

    // Variables to hold the min, max, and average runtimes
    double min_runtime, max_runtime, global_runtime_sum, avg_runtime;

    // Use MPI_Reduce to find the minimum runtime
    MPI_Reduce(&runtime, &min_runtime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    // Use MPI_Reduce to find the maximum runtime
    MPI_Reduce(&runtime, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Use MPI_Reduce to find the sum of runtimes (to calculate the average later)
    MPI_Reduce(&runtime, &global_runtime_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        avg_runtime = global_runtime_sum / size;
        printf("Min Runtime: %f seconds\n", min_runtime);
        printf("Max Runtime: %f seconds\n", max_runtime);
        printf("Avg Runtime: %f seconds\n", avg_runtime);
    }

    for (int i = 0; i < local_X_limit; i++) {
        MPI_Gather(local_grid[i], Y_limit, MPI_INT, global_grid ? global_grid[i] : MPI_IN_PLACE, Y_limit, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
    write_output(global_grid, X_limit, Y_limit, input_file_name, num_of_generations,size);

    // Clean up
    // for (int i = 0; i < local_X_limit; i++) {
    //     delete[] local_grid[i];
    // }
    // delete[] local_grid;

    // delete[] top_ghost_row;
    // delete[] bottom_ghost_row;

    // if (rank == 0) {
    //     for (int i = 0; i < X_limit; i++) {
    //         delete[] global_grid[i];
    //     }
    //     delete[] global_grid;
    // }

    MPI_Finalize();
    return 0;
}

