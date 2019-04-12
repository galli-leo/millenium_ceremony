#ifndef ALGORITHMS_CPP
#define ALGORITHMS_CPP

#include <algorithm>
#include <stack>
#include "algorithms.h"
#include "graph.h"

using namespace std;

#include <algorithm>
#include <iostream>
#include <vector>

list<Node>* eulerian_tour(Graph* graph, Node source)
{
    list<Node>* path = new list<Node>();

    stack<Node> stack;
    stack.push(source);

    while (!stack.empty())
    {
        Node current = stack.top();

        auto nbrs = graph->neighbours(current);

        if (nbrs->size() < 1)
        {
            stack.pop();
            path->push_back(current);
        }
        else
        {
            // Next node
            auto next = nbrs->begin();
            Node next_node = *next;

            // Delete edge between current -> next
            graph->remove_edge(next, current, nbrs);

            // push on stack
            stack.push(next_node);
        }
    }

    return path;
}

int weird_mod(int stelae, int m)
{
    return ((stelae - 1) % m + m) % m + 1;
}

list<Node>* ceremony_tour(Graph* graph, Node source, int level, int player)
{
    list<Node>* path = new list<Node>();

    Node current;

    while (current != source)
    {
        if (current.stelae == -1) current = source;

        list<Node>* neighbours = graph->neighbours(current);
        
        list<Node>::iterator next = neighbours->begin();

        int ideal_v = weird_mod(current.stelae + current.direction(level, player), graph->m);

        Node requested;

        if (current.level == level)
        {
            if (current.isAltar() || !current.correctV(player))
            {
                next = find_if(neighbours->begin(), neighbours->end(), [&level, &player](const Node &arg){
                    return arg.level == level + 1 && (arg.correctV(player) || arg.stelae == 0);
                });

                requested = Node(level + 1, 0);
            }
            else
            {
                /*next = find_if(neighbours->begin(), neighbours->end(), [&ideal_v](const Node &arg){
                    return arg.stelae == ideal_v;
                });*/

                requested = Node(current.level, ideal_v);
                next = find(neighbours->begin(), neighbours->end(), requested);
            }
        }
        else if (current.level == level + 1)
        {
            if (current.isTorch() || !current.correctV(player))
            {
                next = find_if(neighbours->begin(), neighbours->end(), [&level, &player](const Node &arg){
                    return arg.level == level && (arg.correctV(player) || arg.stelae == 0);
                });

                requested = Node(level, 0);
            }
            else
            {
                /*next = find_if(neighbours->begin(), neighbours->end(), [&ideal_v, &current](const Node &arg){
                    return arg.stelae == ideal_v && arg.level == current.level;
                });*/

                requested = Node(current.level, ideal_v);
                next = find(neighbours->begin(), neighbours->end(), requested);
            }
        }

        if (next == neighbours->end())
        {
            next = neighbours->begin();
            cerr << "Did not find the requested edge, level: " << level << ",  current: (" << current.level << ", " << current.stelae << ")" << endl;
            cerr << "\tRequested: (" << requested.level << ", " << requested.stelae << ")" << endl;
            cerr << "\tAvailable: ";

            for (auto node : *neighbours)
            {
                cerr << "(" << node.level << ", " << node.stelae << ") ";
            }

            cerr << endl;
        }

        Node next_node = *next;

        path->push_back(next_node);

        // Delete the edge
        graph->remove_edge(next, current, neighbours);

        current = next_node;
    }

    return path;
}

// Returns size of maximum matching 
int BipGraph::hopcroftKarp() 
{ 
    // pairU[u] stores pair of u in matching where u 
    // is a vertex on left side of Bipartite Graph. 
    // If u doesn't have any pair, then pairU[u] is NIL 
    pairU = new int[m+1]; 
  
    // pairV[v] stores pair of v in matching. If v 
    // doesn't have any pair, then pairU[v] is NIL 
    pairV = new int[n+1]; 
  
    // dist[u] stores distance of left side vertices 
    // dist[u] is one more than dist[u'] if u is next 
    // to u'in augmenting path 
    dist = new int[m+1]; 
  
    // Initialize NIL as pair of all vertices 
    for (int u=0; u<=m; u++) 
        pairU[u] = NIL; 
    for (int v=0; v<=n; v++) 
        pairV[v] = NIL; 
  
    // Initialize result 
    int result = 0; 
  
    // Keep updating the result while there is an 
    // augmenting path. 
    while (bfs()) 
    { 
        // Find a free vertex 
        for (int u=1; u<=m; u++) 
  
            // If current vertex is free and there is 
            // an augmenting path from current vertex 
            if (pairU[u]==NIL && dfs(u)) 
                result++; 
    } 
    return result; 
} 
  
// Returns true if there is an augmenting path, else returns 
// false 
bool BipGraph::bfs() 
{ 
    queue<int> Q; //an integer queue 
  
    // First layer of vertices (set distance as 0) 
    for (int u=1; u<=m; u++) 
    { 
        // If this is a free vertex, add it to queue 
        if (pairU[u]==NIL) 
        { 
            // u is not matched 
            dist[u] = 0; 
            Q.push(u); 
        } 
  
        // Else set distance as infinite so that this vertex 
        // is considered next time 
        else dist[u] = INF; 
    } 
  
    // Initialize distance to NIL as infinite 
    dist[NIL] = INF; 
  
    // Q is going to contain vertices of left side only.  
    while (!Q.empty()) 
    { 
        // Dequeue a vertex 
        int u = Q.front(); 
        Q.pop(); 
  
        // If this node is not NIL and can provide a shorter path to NIL 
        if (dist[u] < dist[NIL]) 
        { 
            // Get all adjacent vertices of the dequeued vertex u 
            list<int>::iterator i; 
            for (i=adj[u].begin(); i!=adj[u].end(); ++i) 
            { 
                int v = *i; 
  
                // If pair of v is not considered so far 
                // (v, pairV[V]) is not yet explored edge. 
                if (dist[pairV[v]] == INF) 
                { 
                    // Consider the pair and add it to queue 
                    dist[pairV[v]] = dist[u] + 1; 
                    Q.push(pairV[v]); 
                } 
            } 
        } 
    } 
  
    // If we could come back to NIL using alternating path of distinct 
    // vertices then there is an augmenting path 
    return (dist[NIL] != INF); 
} 
  
// Returns true if there is an augmenting path beginning with free vertex u 
bool BipGraph::dfs(int u) 
{ 
    if (u != NIL) 
    { 
        list<int>::iterator i; 
        for (i=adj[u].begin(); i!=adj[u].end(); ++i) 
        { 
            // Adjacent to u 
            int v = *i; 
  
            // Follow the distances set by BFS 
            if (dist[pairV[v]] == dist[u]+1) 
            { 
                // If dfs for pair of v also returns 
                // true 
                if (dfs(pairV[v]) == true) 
                { 
                    pairV[v] = u; 
                    pairU[u] = v; 
                    return true; 
                } 
            } 
        } 
  
        // If there is no augmenting path beginning with u. 
        dist[u] = INF; 
        return false; 
    } 
    return true; 
} 
  
// Constructor 
BipGraph::BipGraph(int m, int n) 
{ 
    this->m = m; 
    this->n = n; 
    adj = new list<int>[m+1]; 
} 
  
// To add edge from u to v and v to u 
void BipGraph::addEdge(int u, int v) 
{ 
    adj[u].push_back(v); // Add u to v’s list. 
}

#endif