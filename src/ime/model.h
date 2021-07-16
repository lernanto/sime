/**
 * 输入法模型.
 */

#ifndef _MODEL_H_
#define _MODEL_H_

#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include "log.h"
#include "common.h"
#include "feature.h"


namespace ime
{

/**
 * 输入法模型，支持预测和更新操作.
 *
 * 当前只是稀疏线性模型，只支持普通 SGD 更新
 */
class Model
{
public:
    explicit Model(double lr = 0.01) : weights(), learning_rate(lr) {}

    bool save(std::ostream &os) const;

    bool save(const std::string &fname) const
    {
        std::ofstream os(fname);
        return save(os);
    }

    bool load(std::istream &is);

    bool load(const std::string &fname)
    {
        std::ifstream is(fname);
        return load(is);
    }

    template<typename Iterator>
    double score(Iterator begin, Iterator end) const
    {
        double sum = 0;

        for (auto i = begin; i != end; ++i)
        {
            auto iter = weights.find(i->first);
            if (iter != weights.cend())
            {
                sum += i->second * iter->second;
            }
        }

        return sum;
    }

    double score(const Node &node) const
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

    void compute_score(Node &node) const
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

    template<typename Iterator>
    void update(Iterator begin, Iterator end, double delta)
    {
        for (auto i = begin; i != end; ++i)
        {
            DEBUG << "update: " << i->first << ':' << weights[i->first]
                << " + " << i->second << " * " << delta << " * " << learning_rate
                << " = " << weights[i->first] + i->second * delta * learning_rate << std::endl;
            weights[i->first] += i->second * delta * learning_rate;
        }
    }

    template<typename FeatureIterator, typename DeltaIterator>
    void update(
        FeatureIterator feature_begin,
        FeatureIterator feature_end,
        DeltaIterator delta_begin,
        DeltaIterator delta_end
    )
    {
        FeatureIterator fi;
        DeltaIterator di;
        for (
            fi = feature_begin, di = delta_begin;
            (fi != feature_end) && (di != delta_end);
            ++fi, ++di
        )
        {
            DEBUG << "update: " << *fi << " +" << *di << std::endl;;
            update(fi->begin(), fi->end(), *di);
        }
    }

    std::ostream & output_score(std::ostream &os, const Node &node) const
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

private:
    std::unordered_map<std::string, double> weights;
    double learning_rate;
};

}   // namespace ime

#endif  // _MODEL_H_
