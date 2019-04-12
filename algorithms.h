#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "graph.h"
#include <queue>
#include <list>

using namespace std;

int weird_mod(int stelae, int m);

list<Node>* eulerian_tour(Graph* graph, Node source);

/**
 * Ceremony tour
 */
list<Node>* ceremony_tour(Graph* graph, Node source, int level, int player);

#define NIL 0 
#define INF 2147483647 

// A class to represent Bipartite graph for Hopcroft 
// Karp implementation 
class BipGraph 
{ 
    // m and n are number of vertices on left 
    // and right sides of Bipartite Graph 
    int m, n;
  
public: 
    // adj[u] stores adjacents of left side 
    // vertex 'u'. The value of u ranges from 1 to m. 
    // 0 is used for dummy vertex 
    list<int> *adj; 
  
    // These are basically pointers to arrays needed 
    // for hopcroftKarp() 
    int *pairU, *pairV, *dist; 

    BipGraph(int m, int n); // Constructor 
    void addEdge(int u, int v); // To add edge 
  
    // Returns true if there is an augmenting path 
    bool bfs(); 
  
    // Adds augmenting path if there is one beginning 
    // with u 
    bool dfs(int u); 
  
    // Returns size of maximum matcing 
    int hopcroftKarp(); 
};

#endif