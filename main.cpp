#ifndef MAIN_CPP
#define MAIN_CPP

#include <iostream>
#include <fstream>

#include "graph.h"
#include "algorithms.h"

using namespace std;

static int parseint(void)
{
    int c, n;

    

    n = getchar_unlocked() - '0';
    while ((c = getchar_unlocked()) >= '0')
        n = 10*n + c-'0';

    return n;
}

static void printint(int n)
{
    int buf[10];
    int i = 0;
    while (n > 0) {
        buf[i] = n % 10;
        n = n / 10;
        i++;
    }

    for (int j = i-1; j >= 0; j--) {
        putchar_unlocked(buf[j] + '0');
    }
}

vector<Node>* solve_graph(Graph* graph, Graph* euler, int player = 1)
{
    //int solution_size = (graph->m)*((graph->n*graph->k) - (graph->k+graph->n+2));
    vector<Node> solution;// = new vector<Node>();
    //solution->reserve(solution_size+10);

    for (int l = 0; l < graph->n + 1; l++)
    {
        if (l == 0)
        {
            Node source = Node(0, 0);
            // get path via ceremony tour
            while (solution.size() < 3 * (graph->m / 2)) 
            {
                auto path = ceremony_tour(graph, source, 0, player);
                solution.insert(solution.end(), path->begin(), path->end());
            }
        }
        else
        {
            //auto end = solution->rend();
            for (int i = solution.size() - 1; i >= 0; i--)
            {
                //Node v1 = solution[i];

                if (solution[i].level == l && graph->neighbours(solution[i])->size() > 0)
                {
                    auto path = ceremony_tour(graph, solution[i], l, player);

                    solution.insert(solution.begin() + i + 1, path->begin(), path->end());
                }
            }
        }
    }

    if (euler != nullptr)
    {
        //auto end = solution->rend();
        for (int i = solution.size() - 1; i >= 0; i--)
        {
            Node v1 = solution[i];

            if (euler->neighbours(v1)->size() > 0)
            {
                auto path = eulerian_tour(euler, v1);

                Node next = *path->begin();

                auto beginning = path->begin();

                if (next == v1)
                {
                    //cerr << "Cannot have loops in our graph for node: " << "(" << v1.level << ", " << v1.stelae << ")" << endl;
                    // We have the source node as well here, just remove it
                    beginning = std::next(beginning);
                }

                solution.insert(solution.begin() + i + 1, beginning, path->end());
            }
        }
    }

    for (auto node : solution)
    {
        printf("%d %d\n", node.level, node.stelae);
        //cout << node.level << " " << node.stelae << "\n";
    }

    return nullptr;
}


pair<Graph*, Graph*> read_graph(istream& input, int n, int m, int k)
{
    Graph* graph = new Graph(n, m, k);
    //Graph* graph2 = new Graph(n, m, k);
    Graph* euler = nullptr;
    //Graph* euler2 = nullptr;
    if (k > 1) 
    {
        euler = new Graph(n, m, k-1);
        //euler2 = new Graph(n, m, k-1);
    }

    Node altar = Node(0, 0);
    Node torch = Node(n+1, 0);

    //graph->add_node(altar);
    //graph->adj[graph->adj_idx(altar)].reserve(m);
    //graph->add_node(torch);
    //graph->adj[graph->adj_idx(torch)].reserve(m);

    // The last layer does not have any inter layer ropes, so we do not need this loop for that.
    for (int l = 1; l < n; l++) {
        BipGraph matcher(m, m);
        for (int s = 1; s <= m; s++) {
            //graph->adj[graph->adj_idx(Node(l, s))].reserve(6);
            //graph2->adj[graph->adj_idx(Node(l, s))].reserve(6);
            if (k > 1)
            {
                //euler->adj[graph->adj_idx(Node(l, s))].reserve((k)*2);
                //euler2->adj[graph->adj_idx(Node(l, s))].reserve((k)*2);
            }
            for (int r = 0; r < k; r++) {
                int other_s;// = parseint();
                input >> other_s;
                if (k > 1)
                {
                    matcher.addEdge(s, other_s);
                }
                else
                {
                    graph->add_edge(Node(l, s), Node(l+1, other_s));
                    //graph2->add_edge(Node(l, s), Node(l+1, other_s));
                }
            }
            graph->add_edge(Node(l, s), Node(l, weird_mod(s + 1, m)));
            //graph2->add_edge(Node(l, s), Node(l, weird_mod(s + 1, m)));
        }
        if (k > 1)
        {
            matcher.hopcroftKarp();
            for (int s = 1; s <= m; s++)
            {
                int matched = matcher.pairU[s];
                if (matched == NIL)
                {
                    cerr << "Error: could not match vertex: " << "(" << l << ", " << s << ")" << endl;
                }
                
                for (auto other_s : matcher.adj[s])
                {
                    if (other_s == matched)
                    {
                        graph->add_edge(Node(l, s), Node(l+1, other_s));
                        //graph2->add_edge(Node(l, s), Node(l+1, other_s));
                    }
                    else
                    {
                        euler->add_edge(Node(l, s), Node(l+1, other_s));
                        //euler2->add_edge(Node(l, s), Node(l+1, other_s));
                    }
                }
            }
        }
    }

    for (int s = 1; s <= m; s++) {
        graph->add_edge(altar, Node(1, s));
        graph->add_edge(Node(n, s), torch);
        graph->add_edge(Node(n, s), Node(n, weird_mod(s + 1, m)));
        /*graph2->add_edge(altar, Node(1, s));
        graph2->add_edge(Node(n, s), torch);
        graph2->add_edge(Node(n, s), Node(n, weird_mod(s + 1, m)));*/
    }

    return make_pair(graph, euler);
}

void solve_testcase(istream& input)
{
    int n, m, k;
    input >> n;
    input >> m;
    input >> k;

    //cout << "Variables: " << n << " " << m << " " << k << endl;

    auto graphs = read_graph(input, n, m, k);
    Graph* graph = graphs.first;
    //cout << "Graph:" << endl;
    //graph->print_adj();
    Graph* euler = graphs.second;
    //cout << "Euler:" << endl;
    //euler->print_adj();
    Graph* euler2 = nullptr;//graphs.second.second;

    if (euler != nullptr) {
        euler2 = euler->copy();
    }
    Graph* graph2 = graph->copy();//graphs.second.first;

    auto solution1 = solve_graph(graph, euler);

    //char buffer [solution1->size()*20];

    delete graph;
    if (euler != nullptr)
    {
        delete euler;
    }

#if LOCAL
    cout << "Solution 2:" << endl;
    graph2->print_adj();
#endif

    auto solution2 = solve_graph(graph2, euler2, 2);

    delete graph2;
    if (euler2 != nullptr)
    {
        delete euler2; 
    }
}

void solve(istream& input)
{
    int number_testcases;
    input >> number_testcases;

    for (int t = 0; t < number_testcases; t++)
    {
        // It always works :P
        cout << "Case " << t+1 << ": Yes" << endl;

        solve_testcase(input);
    }
}

int main() 
{
    #ifdef LOCAL
    ifstream infile("testcase.txt");
    solve(infile);
    #else
    solve(cin);
    #endif
    return 0;
}

#endif