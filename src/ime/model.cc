/**
 *
 */

#include <string>
#include <iostream>
#include <sstream>

#include "model.h"
#include "log.h"


namespace ime
{

bool Model::save(std::ostream &os) const
{
    for (auto & i : weights)
    {
        os << i.first << '\t' << i.second << std::endl;;
    }

    INFO << weights.size() << " features saved" << std::endl;
    return true;
}

bool Model::load(std::istream &is)
{
    weights.clear();

    while (!is.eof())
    {
        std::string line;
        std::string feature;
        double weight;

        std::getline(is, line);
        std::stringstream ss(line);
        ss >> feature >> weight;
        if (!feature.empty())
        {
            DEBUG << "load feature " << feature << ", weight = " << weight << std::endl;
            weights.emplace(feature, weight);
        }
    }

    INFO << weights.size() << " features loaded" << std::endl;
    return true;
}

}   // namespace ime
