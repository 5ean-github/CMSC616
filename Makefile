all: serial life-nonblocking

# Compiler for serial
CXX = g++
CXXFLAGS = -std=c++11 -O2

# Compiler for MPI
MPICXX = mpicxx
MPICXXFLAGS = -std=c++11 -O2

# Target for serial version
serial: serial.cpp
	$(CXX) $(CXXFLAGS) -o life $<

# Target for MPI nonblocking version
life-nonblocking: life-nonblocking.cpp
	$(MPICXX) $(MPICXXFLAGS) -o life-nonblocking $<

# Clean target to remove executables
clean:
	rm -f serial.o life life-nonblocking
