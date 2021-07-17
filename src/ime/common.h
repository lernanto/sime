/**
 * 公共数据结构.
 */

#ifndef _COMMON_H_
#define _COMMON_H_

#include <cmath>
#include <cassert>
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
    const Word *word;
    /// 指向路径中前一个有词的节点，用于构造 n-gram 特征
    const Node *prev_word;
    /// 局部特征，对经过节点的所有路径都生效的特征保存在这里
    std::vector<std::pair<std::string, double>> local_features;
    /// 全局特征，描述整条路径的特征，只有当节点是路径的最后一个节点才生效
    std::vector<std::pair<std::string, double>> global_features;
    /// 为加速计算，保存该节点及之前子路径局部特征的得分
    double local_score;
    /// 以该节点为代表的路径（即以该节点结尾的路径）的得分
    double score;

    Node() :
        prev(nullptr),
        code_pos(0),
        text_pos(0),
        word(nullptr),
        prev_word(nullptr),
        local_features(),
        global_features(),
        local_score(0),
        score(0) {}

    Node(const Node *prev_) :
        prev(prev_),
        code_pos(prev_->code_pos),
        text_pos(prev_->text_pos),
        word(nullptr),
        prev_word((prev_->word != nullptr) ? prev_ : prev_->prev_word),
        local_features(),
        global_features(),
        local_score(0),
        score(0)
        {
            assert(prev_ != nullptr);
        }

    Node(
        const Node *prev_,
        size_t code_pos_,
        size_t text_pos_,
        const Word *word_
    ) :
        prev(prev_),
        code_pos(code_pos_),
        text_pos(text_pos_),
        word(word_),
        prev_word((prev_->word != nullptr) ? prev_ : prev_->prev_word),
        local_features(),
        global_features(),
        local_score(0),
        score(0)
        {
            assert(prev_ != nullptr);
        }

    Node(const Node &other) :
        prev(other.prev),
        code_pos(other.code_pos),
        text_pos(other.text_pos),
        word(other.word),
        prev_word(other.prev_word),
        local_features(other.local_features),
        global_features(other.global_features),
        local_score(other.local_score),
        score(other.score) {}

    Node(Node &&other) :
        prev(other.prev),
        code_pos(other.code_pos),
        text_pos(other.text_pos),
        word(other.word),
        prev_word(other.prev_word),
        local_features(std::move(other.local_features)),
        global_features(std::move(other.global_features)),
        local_score(other.local_score),
        score(other.score) {}

    bool operator > (const Node &other) const
    {
        return score > other.score;
    }
};

inline void swap(Node &a, Node &b)
{
    std::swap(a.prev, b.prev);
    std::swap(a.code_pos, b.code_pos);
    std::swap(a.text_pos, b.text_pos);
    std::swap(a.word, b.word);
    std::swap(a.local_features, b.local_features);
    std::swap(a.global_features, b.global_features);
    std::swap(a.local_score, b.local_score);
    std::swap(a.score, b.score);
}

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
