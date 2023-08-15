/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;

ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/

// Calculating maximum node for a level
__global__ void find_Level(int *d_csrList, int *d_offset, int *d_max_Index, int s_Level, int e_Level){
    
    // c_node is present node
    int c_node = blockIdx.x * blockDim.x + threadIdx.x + s_Level;  
    
    if(c_node <= e_Level){
        // checking max node 
        if(d_csrList[d_offset[c_node+1]-1] > d_max_Index[0]){
            atomicMax(&d_max_Index[0], d_csrList[d_offset[c_node+1]-1]);
        }
    }

}
       
// Calculating indegree of each node in a level      
__global__ void in_Degree(int *d_csrList, int *d_offset, int *d_aid, int s_Level, int e_Level, int *d_active_node){
    
    // c_node is present node
    int c_node = blockIdx.x * blockDim.x + threadIdx.x + s_Level;

    if(c_node <= e_Level){
        int len = d_offset[c_node+1] - d_offset[c_node];
        // if edge from node is active increment indegree of present node
        if(d_active_node[c_node]){
            for(int i=0; i<len; i++){
                atomicAdd(&d_aid[d_csrList[d_offset[c_node]+i]], 1);
            }
        }
    }

}
    
// Calculating node is active or not in each level    
__global__ void active_Degree(int *d_aid, int s_Level, int e_Level, int *d_active_node, int *d_apr){
    
    // c_node is present node
    int c_node = blockIdx.x * blockDim.x + threadIdx.x + s_Level;

    if(c_node <= e_Level){
        // comparing apr and d_aid to check node is active or not
        if(d_aid[c_node] >= d_apr[c_node]){
            d_active_node[c_node] = 1;
        }    
    }

}   

// Checking for deactivation rule 2 at each level
__global__ void d_active_Degree(int s_Level, int e_Level, int *d_active_node, int *d_apr, int V){
    
    // c_node is present node
    int c_node = blockIdx.x * blockDim.x + threadIdx.x + s_Level;

    if(c_node <= e_Level){
        // Checking left and right node is active or not
        if(c_node > s_Level && c_node < e_Level){
            // If left and right node are inactive present node is set to inactive
            if(d_active_node[c_node-1]==0 && d_active_node[c_node+1]==0){
                d_active_node[c_node] = 0;
            }  
        } 
    } 

}   

// Calculating active nodes at each level
__global__ void active_nodes_Level(int s_Level, int e_Level, int *d_active_node, int *d_cnt){
    
    // c_node is present node
    int c_node = blockIdx.x * blockDim.x + threadIdx.x + s_Level;

    if(c_node <= e_Level){
          if(d_active_node[c_node]){
              atomicAdd(&d_cnt[0], 1);   // Adds 1 to d_cnt for each active node
          }
    }
}
    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    // int *d_activeVertex;
	  // cudaMalloc(&d_activeVertex, L*sizeof(int));

    /***Important***/

    // Initialize d_aid array to zero for each vertex
    // Make sure to use comments

    /***END***/
    double starttime = rtclock(); 

    /*********************************CODE AREA*****************************************/

    //Variables on host  
    int *h_max_Index;     // max node at a level
    int *level_Start;     // array stores start index of each level
    int *level_End;       // array stores end index of each level
    int *h_active_node;   // array stores active or not of each node
    int *h_cnt;           // Number of active nodes at each level 

    //Allocating memory on host
    h_cnt = (int*)malloc(sizeof(int));
    h_max_Index = (int*)malloc(sizeof(int));
    level_Start = (int*)malloc(L*sizeof(int));
    level_End = (int*)malloc(L*sizeof(int));
    h_active_node = (int*)malloc(V*sizeof(int));

    memset(h_active_node, 0, V*sizeof(int)); // setting initially all to zero

    //Variables on devide
    int *d_max_Index;
    int *d_active_node;
    int *d_cnt;

    //Allocating memory on device and 
    cudaMalloc(&d_cnt, sizeof(int));
    cudaMalloc(&d_max_Index, sizeof(int));  
    cudaMalloc(&d_active_node, V*sizeof(int));
    
    // setting initially all to zero
    cudaMemset(d_max_Index, 0, sizeof(int));
    cudaMemset(d_aid, 0, V*sizeof(int));
    cudaMemset(d_active_node, 0, V*sizeof(int));

    int zero_Level_count = 0; // for number of nodes in level zero
    int threads = 1024;       // Number of threads in launching the kernel
    int block;                // Number of blocks in launching the kernel
    
    // Finding number of nodes at level zero
    for(int i=0; i<V; i++){
        if(h_apr[i] == 0){
            zero_Level_count++;
        }
        else{
            break;
        }
    }

    // setting start and end node of level zero
    level_Start[0] = 0;
    level_End[0] = zero_Level_count-1;
    
    // setting active node of level zero to 1
    for(int i=0; i<=level_End[0]; i++){
        h_active_node[i] = 1;
    }
    //copy the h_active_node array to device
    cudaMemcpy(d_active_node, h_active_node, V*sizeof(int), cudaMemcpyHostToDevice);

    // Finding start and end node of each level
    for(int i=0; i<L-1; i++){

        // Number of blocks for kernel launch
        block = ceil((float)(level_End[i]-level_Start[i]+1)/1024); 
        // Finding max node at level
        find_Level<<<block, threads>>>(d_csrList, d_offset, d_max_Index, level_Start[i], level_End[i]);
        //copy the d_max_Index array to host
        cudaMemcpy(h_max_Index, d_max_Index, sizeof(int), cudaMemcpyDeviceToHost);
        // setting start and end node of level
        level_Start[i+1] = level_End[i]+1;
        level_End[i+1] = h_max_Index[0];

    }

    // Finding node is active or not 
    for(int i=0; i<L-1; i++){

        block = ceil((float)(level_End[i]-level_Start[i]+1)/1024);   // Number of blocks for kernel launch
        // Kernel call finds indegree of nodes
        in_Degree<<<block, threads>>>(d_csrList, d_offset, d_aid, level_Start[i], level_End[i], d_active_node);
        cudaDeviceSynchronize();

        // Number of blocks for kernel launch
        block = ceil((float)(level_End[i+1]-level_Start[i+1]+1)/1024);  
        // Kernel call finds active or not of nodes by comparing apr and d_indegre
        active_Degree<<<block, threads>>>(d_aid, level_Start[i+1], level_End[i+1], d_active_node, d_apr);
        cudaDeviceSynchronize();
        
        //Kernel call checks for left and right active and sets 
        d_active_Degree<<<block, threads>>>(level_Start[i+1], level_End[i+1], d_active_node, d_apr, V);
        cudaDeviceSynchronize();

    }   
    
    // Calculating active nodes in the level
    for(int i=0; i<L; i++){
        // setting initially to zero before each kernel call
        cudaMemset(d_cnt, 0, sizeof(int)); 
        // Number of blocks for kernel launch
        block = ceil((float)(level_End[i]-level_Start[i]+1)/1024);
        // Kernel gives active nodes in the level
        active_nodes_Level<<<block, threads>>>(level_Start[i], level_End[i], d_active_node, d_cnt);
        //copy the d_cnt array to host
        cudaMemcpy(h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        // setting h_cnt in h_activeVertex
        h_activeVertex[i] = h_cnt[0];
    }
        

    /********************************END OF CODE AREA**********************************/
    double endtime = rtclock();  
    printtime("GPU Kernel time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    char outFIle[30] = "./output.txt" ;
    printResult(h_activeVertex, L, outFIle);
    if(argc>2)
    {
        for(int i=0; i<L; i++)
        {
            printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
        }
    }

    return 0;
}
