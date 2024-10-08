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
    output_file.open(input_file_name + "." + to_string(num_of_generations) + 
                    ".csv");
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
    for (int i = 0; i < local_X_limit; i++) {
        local_grid[i] = new int[Y_limit]();
    }

    int **previous_life = new int *[local_X_limit];
    for (int i = 0; i < local_X_limit; i++) {
        previous_life[i] = new int[Y_limit+2];
        for (int j = 0; j < Y_limit+2; j++) {
            previous_life[i][j] = 0;
        }
    }


    int** global_grid = nullptr;
    int* global_grid_1D = nullptr;
    if (rank == 0) {
        global_grid = new int*[X_limit];
        for (int i = 0; i < X_limit; i++) {
            global_grid[i] = new int[Y_limit]();
        }
        read_input_file(global_grid, input_file_name);
        global_grid_1D = new int [X_limit*Y_limit];
        for (int i=0;i<X_limit;i++){
            for (int j=0;j<Y_limit;j++){
                global_grid_1D[i*Y_limit+j] = global_grid[i][j];
            }
        }
    }

    int* local_grid_1D = new int[local_X_limit*Y_limit];

    MPI_Scatter(
        global_grid_1D,             
        local_X_limit*Y_limit,      
        MPI_INT,                    
        local_grid_1D,              
        local_X_limit*Y_limit,      
        MPI_INT,                    
        0,                          
        MPI_COMM_WORLD              
    );

    for (int i=0;i<local_X_limit;i++){
        for (int j=0;j<Y_limit;j++){
            local_grid[i][j] = local_grid_1D[i*Y_limit+j];
        }
    }

    MPI_Request reqs[4];
    int* top_ghost_row = new int[Y_limit]();
    int* bottom_ghost_row = new int[Y_limit]();

    double start_time = MPI_Wtime();

    for (int numg = 0; numg < num_of_generations; numg++) {
        int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1; 

        MPI_Irecv(top_ghost_row, Y_limit, MPI_INT, up, 0, MPI_COMM_WORLD, &reqs[0]); 
        MPI_Irecv(bottom_ghost_row, Y_limit, MPI_INT, down, 0, MPI_COMM_WORLD, &reqs[1]); 

        MPI_Isend(local_grid[0], Y_limit, MPI_INT, up, 0, MPI_COMM_WORLD, &reqs[2]); 
        MPI_Isend(local_grid[local_X_limit - 1], Y_limit, MPI_INT, down, 0, MPI_COMM_WORLD, &reqs[3]);

        //compute local_grid[1:local_X_limit-1]

        for (int i = 0; i < local_X_limit; i++) {
            for (int j = 0; j < Y_limit; j++) {
                previous_life[i][j + 1] = local_grid[i][j];
            }
        }

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
        
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

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
        
    }
    double end_time = MPI_Wtime();
    double runtime = end_time - start_time;

    double min_runtime, max_runtime, global_runtime_sum, avg_runtime;
    MPI_Reduce(&runtime, &min_runtime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&runtime, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&runtime, &global_runtime_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        avg_runtime = global_runtime_sum / size;
        cout << "TIME: Min: " << min_runtime << " s Avg: " << avg_runtime << " s Max: " << max_runtime << " s\n";
    }

    for (int i=0;i<local_X_limit;i++){
        for (int j=0;j<Y_limit;j++){
            local_grid_1D[i*Y_limit+j] = local_grid[i][j];
        }
    }

    MPI_Gather(
        local_grid_1D,              
        local_X_limit*Y_limit,      
        MPI_INT,                    
        global_grid_1D,             
        local_X_limit*Y_limit,      
        MPI_INT,                    
        0,                          
        MPI_COMM_WORLD              
    );

    if (rank==0){
        for (int i=0;i<X_limit;i++){
            for (int j=0;j<Y_limit;j++){
                global_grid[i][j] = global_grid_1D[i*Y_limit+j];
            }
        }
        write_output(global_grid, X_limit, Y_limit, input_file_name, num_of_generations,size);
    }
    
    for (int i = 0; i < local_X_limit; i++) {
        delete[] local_grid[i];
    }
    delete[] local_grid;
    delete[] local_grid_1D;
    for (int i = 0; i < local_X_limit; i++) {
        delete[] previous_life[i];
    }
    delete[] previous_life;

    delete[] top_ghost_row;
    delete[] bottom_ghost_row;

    if (rank == 0) {
        for (int i = 0; i < X_limit; i++) {
            delete[] global_grid[i];
        }
        delete[] global_grid;
        delete[] global_grid_1D;
    }

    MPI_Finalize();
    return 0;
}

