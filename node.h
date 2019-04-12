#ifndef NODE_H
#define NODE_H

#include <vector>
#include <list>

using namespace std;

/**
 * A struct representing a node of our graph.
 */
struct Node {
    Node()
    {

    }

    Node(int level_, int stelae_)
    {
        level = level_;
        stelae = stelae_;
    }

    bool isAltar() const
    {
      return this->level == 0 && this->stelae == 0;
    }

    bool isTorch() const
    {
      return this->level != 0 && this->stelae == 0;
    }

    bool correctV(int player) const
    {
      return player == 1 ? this->stelae % 2 == this->level % 2 : this->stelae % 2 != this->level % 2;
    }

    int direction(int level, int player) const 
    {
      if (player == 1)
      {
        if (level % 2 == this->level % 2)
        {
          return this->stelae % 2 == 0 ? 1 : -1;
        }

        return this->stelae % 2 == 0 ? -1 : 1;
      }

      if (level % 2 == this->level % 2)
      {
        return this->stelae % 2 == 0 ? -1 : 1;
      }

      return this->stelae % 2 == 0 ? 1 : -1;
    }

    int level = -1;
    int stelae = -1;
    vector<Node>::iterator inverse_edge;
};

inline bool operator==(const Node& lhs, const Node& rhs)
{
    return lhs.level == rhs.level && lhs.stelae == rhs.stelae;
}

inline bool operator!=(const Node& lhs, const Node& rhs)
{
    return !(lhs == rhs);
}

namespace std {

  template <>
  struct hash<Node>
  {
    std::size_t operator()(const Node& k) const
    {
      using std::size_t;
      using std::hash;
      using std::string;

      // Compute individual hash values for first,
      // second and third and combine them using XOR
      // and bit shifting:

      return ((hash<int>()(k.level)
               ^ (hash<int>()(k.stelae) << 1)));
    }
  };

}
#endif