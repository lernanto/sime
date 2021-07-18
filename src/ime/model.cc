/**
 *
 */

#include <string>
#include <iostream>
#include <sstream>

#include "model.h"
#include "log.h"
#include "common.h"


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

double Model::score(const Node &node) const
{
    double sum = 0;

    for (auto p = &node; p != nullptr; p = p->prev)
    {
        for (auto &f : p->local_features)
        {
            auto iter = weights.find(f.first);
            if (iter != weights.cend())
            {
                sum += f.second * iter->second;
            }
        }
    }

    for (auto &f : node.global_features)
    {
        auto iter = weights.find(f.first);
        if (iter != weights.cend())
        {
            sum += f.second * iter->second;
        }
    }

    return sum;
}

void Model::compute_score(Node &node) const
{
    // 因为是线性模型且特征是子路径局部特征的超集，从前一个节点取局部特征分数以加速计算
    node.local_score = (node.prev != nullptr) ? node.prev->local_score : 0;

    // 累加本节点局部特征的得分
    for (auto &f : node.local_features)
    {
        auto iter = weights.find(f.first);
        if (iter != weights.cend())
        {
            node.local_score += f.second * iter->second;
        }
    }

    // 再加上本节点（代表的路径）特有的全局特征
    node.score = node.local_score;
    for (auto &f : node.global_features)
    {
        auto iter = weights.find(f.first);
        if (iter != weights.cend())
        {
            node.score += f.second * iter->second;
        }
    }
}

std::ostream & Model::output_score(std::ostream &os, const Node &node) const
{
    for (auto p = &node; p != nullptr; p = p->prev)
    {
        for (auto &f : p->local_features)
        {
            os << f.first << ':' << f.second << " * ";
            auto iter = weights.find(f.first);
            os << ((iter != weights.cend()) ? iter->second : 0) << " + ";
        }
    }
    for (auto &f : node.global_features)
    {
        os << f.first << ':' << f.second << " * ";
        auto iter = weights.find(f.first);
        os << ((iter != weights.cend()) ? iter->second : 0) << " + ";
    }

    return os;
}

}   // namespace ime
