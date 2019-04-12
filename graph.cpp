#ifndef GRAPH_CPP
#define GRAPH_CPP

#include <iostream>
#include <sstream>
#include "graph.h"

//! Efficiently remove an element from a vector without
//! preserving order. If the element is not the last element
//! in the vector, transfer the last element into its position
//! using a move if possible.
//! Regardless, we then shrink the size of the vector deleting
//! the element at the end, which will either be destructed or
//! the element we were deleting.
//! @note: Effectively invalidates the current iterator.
template<class ValueType>
bool unstable_remove(
    typename std::vector<ValueType>* container,
    typename std::vector<ValueType>::iterator it
    )
{
    // Leave in-situ if we are already the tail element.
    auto lastEl = container->end() - 1;
    if (it != lastEl) {
        // overwrite this element with what is in the last,
        // which should have the same effect as deleting this.
        *it = std::move(*lastEl);
    }
    // release the last cell of the vector, because it should
    // now either be destructed or contain the value we were
    // deleting.
    container->pop_back();

    return true;
}

Graph::Graph(int layers, int stelae, int inter)
{
    n = layers;
    m = stelae;
    k = inter;

    number_nodes = n*m + 2;

    //adj = new unordered_map<Node, vector<Node*>>();
}

/*int Graph::index(Node node)
{
    return this->index(node.level, node.stelae);
}

int Graph::index(int level, int stelae)
{
    return indices[level * m + stelae];
}

void Graph::set_index(Node node, int index)
{
    this->set_index(node.level, node.stelae, index);
}

void Graph::set_index(int level, int stelae, int index)
{
    indices[level * m + stelae] = index;
}*/

void Graph::add_edge(Node v, Node u)
{
    //this->add_node(v);
    //this->add_node(u);

    this->adj[v].push_back(u);
    //this->degrees[v] = this->degrees[v]+1;
    this->adj[u].push_back(v);
    this->adj[v].rbegin()->inverse_edge = std::next(this->adj[u].rbegin()).base();
    this->adj[u].rbegin()->inverse_edge = std::next(this->adj[v].rbegin()).base();
    //this->degrees[u] = this->degrees[u]+1;
}

void Graph::remove_edge(Node v, Node u, list<Node>* nbrs = nullptr)
{
    if (nbrs == nullptr) nbrs = this->neighbours(v);
    this->remove_edge(find(nbrs->begin(), nbrs->end(), v), u, nbrs);
}

void Graph::remove_edge(list<Node>::iterator v, Node u, list<Node>* nbrs = nullptr) 
{
    Node v_node = *v;
    if (nbrs == nullptr) nbrs = this->neighbours(v_node);
    //unstable_remove<Node>(nbrs, v);
    nbrs->erase(v);
    auto other_nbrs = this->neighbours(v_node);
    other_nbrs->erase(v_node.inverse_edge);
    //other_nbrs->erase(find(other_nbrs->begin(), other_nbrs->end(), u));
    //unstable_remove<Node>(other_nbrs, find(other_nbrs->begin(), other_nbrs->end(), u));
    //this->degrees[v_node] = this->degrees[v_node]-1;
    //this->degrees[u] = this->degrees[u]-1;
}

int Graph::degree(Node v)
{
    return this->degrees[v];
}

const Node* Graph::add_node(int layer, int stelae)
{
    return this->add_node(Node(layer, stelae));
}

const Node* Graph::add_node(Node node)
{
    /*this->nodes.insert(node);
    return &*this->nodes.find(node);*/
    return nullptr;
}

list<Node>* Graph::neighbours(Node node)
{
    return &(this->adj[node]);
}

Graph* Graph::copy()
{
    Graph* copy = new Graph(this->n, this->m, this->k);

    for (auto v : this->adj)
    {
        for (auto u : v.second)
        {
            copy->add_edge(v.first, u);
        }
    }
    //copy->degrees = unordered_map<Node, int>(this->degrees);
    //copy->nodes = unordered_set<Node>(this->nodes);

    return copy;
}

void Graph::print_adj(ostream &output)
{
    for (auto& pair : this->adj) {
        auto node = pair.first;
        output << "(" << node.level << ", " << node.stelae << ") => ";
        for (auto& neigh : pair.second) {
            output << "(" << neigh.level << ", " << neigh.stelae << "), ";
        }

        output << endl;
    }
}

string Graph::to_string()
{
    stringstream out;
    this->print_adj(out);
    return out.str();
}

#endif