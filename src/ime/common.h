/**
 * 公共数据结构.
 */

#ifndef _COMMON_H_
#define _COMMON_H_

#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <iostream>


namespace ime
{

/**
 * 代表词典中一个词的信息.
 */
struct Word
{
    std::string code;
    std::string text;

    Word(const std::string &code_, const std::string &text_) :
        code(code_), text(text_) {}
};

/**
 * 集束搜索中一个集束中的节点，同时也是输出结果路径中的节点.
 */
struct Node
{
    /// 指向路径中前一个节点
    const Node *prev;
    size_t code_pos;
    size_t text_pos;
    std::string code;
    const Word *word;
    /// 指向路径中前一个有词的节点，用于构造 n-gram 特征
    const Node *prev_word;
    /// 局部特征，对经过节点的所有路径都生效的特征保存在这里
    std::vector<std::pair<std::string, double>> local_features;
    /// 全局特征，描述整条路径的特征，只有当节点是路径的最后一个节点才生效
    std::vector<std::pair<std::string, double>> global_features;
    double score;

    Node() :
        prev(nullptr),
        code_pos(0),
        text_pos(0),
        code(),
        word(nullptr),
        prev_word(nullptr),
        local_features(),
        global_features(),
        score(0) {}

    Node(const Node &other) :
        prev(other.prev),
        code_pos(other.code_pos),
        text_pos(other.text_pos),
        code(other.code),
        word(other.word),
        prev_word(nullptr),
        local_features(other.local_features),
        global_features(other.global_features),
        score(other.score) {}

    bool operator > (const Node &other) const
    {
        return score > other.score;
    }
};

/**
 * 用于记录训练和预测过程中的一些统计量.
 */
class Metrics
{
public:
    double get(const std::string &key) const
    {
        auto iter = data.find(key);
        if (iter != data.cend())
        {
            return iter->second;
        }
        else
        {
            return NAN;
        }
    }

    void set(const std::string &key, double val)
    {
        data.emplace(key, val);
    }

    void clear()
    {
        data.clear();
    }

    std::map<std::string, double>::const_iterator begin() const
    {
        return data.begin();
    }

    std::map<std::string, double>::const_iterator end() const
    {
        return data.end();
    }

private:
    std::map<std::string, double> data;
};

inline std::ostream & operator << (std::ostream &os, const Word &word)
{
    return os << word.text << '(' << word.code << ')';
}

inline std::ostream & operator << (std::ostream &os, const Metrics &metrics)
{
    for (auto &i : metrics)
    {
        os << i.first << " = " << i.second << ", ";
    }
    return os;
}

}   // namespace

#endif  // _COMMONT_H_
