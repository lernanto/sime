/**
 * 公共数据结构.
 */

#ifndef _COMMON_H_
#define _COMMON_H_

#include <string>
#include <vector>
#include <utility>


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

}   // namespace

#endif  // _COMMONT_H_
