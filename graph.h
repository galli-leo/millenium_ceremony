#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <list>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include "node.h"

using namespace std;

/*
 *
 */
class Graph {
    public:
        // Properties
        int n = 0, m = 0, k = 0;
        int number_nodes;

        //unordered_map<Node, list<Node> > adj;
        list<Node>* adj;
        unordered_map<Node, int> degrees;
        //unordered_set<Node> nodes;

        // Methods
        Graph(int n, int m, int k);

        //int index(Node node);
        //int index(int level, int stelae);

        const Node* add_node(Node node);
        const Node* add_node(int level, int stelae);

        list<Node>* neighbours(Node node);
        int degree(Node node);

        void add_edge(Node v, Node u);

        void remove_edge(Node v, Node u, list<Node>* nbrs);
        void remove_edge(list<Node>::iterator v, Node u, list<Node>* nbrs);

        void print_adj(ostream &output = cout);

        int adj_idx(Node node);

        string to_string();

        Graph* copy();

    private:
        // Properties
        //int current_index = -1;

        // Hacky
        //Node first_node;

        //void set_index(Node node, int index);
        //void set_index(int level, int stelae, int index);
};

#endif