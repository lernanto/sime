/**
 * 基于结构化感知机（structured perceptron）的输入法引擎.
 */

#ifndef _IME_H_
#define _IME_H_

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>

#include "log.h"


namespace ime
{

struct Word
{
    std::string code;
    std::string text;

    Word(const std::string &code_, const std::string &text_) :
        code(code_), text(text_) {}
};

class Dictionary
{
public:
    explicit Dictionary(const std::string &fname) : data()
    {
        load(fname);
    }

    bool load(std::istream &is);

    bool load(const std::string &fname)
    {
        std::ifstream is(fname);
        return load(is);
    }

    void find(
        const std::string &code,
        std::multimap<std::string, Word>::const_iterator &begin,
        std::multimap<std::string, Word>::const_iterator &end
    ) const
    {
        auto range = data.equal_range(code);
        begin = range.first;
        end = range.second;
    }

private:
    std::multimap<std::string, Word> data;
};

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

    template<typename Iterator>
    void update(Iterator begin, Iterator end, double delta)
    {
        if (LOG_LEVEL <= LOG_DEBUG)
        {
            DEBUG << "update: ";
            for (auto i = begin; i != end; ++i)
            {
                DEBUG << i->first << ':' << i->second << ',';
            }
            DEBUG << " +" << delta << '*' << learning_rate << std::endl;
        }

        for (auto i = begin; i != end; ++i)
        {
            DEBUG << i->first << ':' << weights[i->first] << " + " << i->second << '*' << delta << '*' << learning_rate;
            weights[i->first] += i->second * delta * learning_rate;
            DEBUG << " = " << weights[i->first] << std::endl;
        }
    }

    template<typename F>
    void update(
        const std::vector<F> &features,
        const std::vector<double> &deltas
    )
    {
        assert(features.size() == deltas.size());
        for (size_t i = 0; i < features.size(); ++i)
        {
            update(features[i].begin(), features[i].end(), deltas[i]);
        }
    }

    void update(const Node &node, double delta)
    {
        for (auto p = &node; p != nullptr; p = p->prev)
        {
            for (auto &f : p->local_features)
            {
                DEBUG << f.first << ':' << weights[f.first]
                    << " + " << f.second << " * " << delta << " * " << learning_rate
                    << " = " << weights[f.first] + f.second * delta * learning_rate << std::endl;
                weights[f.first] += f.second * delta * learning_rate;
            }
        }

        for (auto &f : node.global_features)
        {
            DEBUG << f.first << ':' << weights[f.first]
                << " + " << f.second << " * " << delta << " * " << learning_rate
                << " = " << weights[f.first] + f.second * delta * learning_rate << std::endl;
            weights[f.first] += f.second * delta * learning_rate;
        }
    }

    void update(const std::vector<Node> &beam, const std::vector<double> &deltas)
    {
        assert(beam.size() == deltas.size());
        for (size_t i = 0; i < beam.size(); ++i)
        {
            update(beam[i], deltas[i]);
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

class Decoder
{
public:
    Decoder(
        const Dictionary &dict_,
        size_t beam_size_ = 20
    ) : dict(dict_), beam_size(beam_size_), model() {}

    bool decode(
        const std::string &code,
        const std::string &text,
        std::vector<std::vector<Node>> &beams,
        size_t beam_size
    ) const;

    bool decode(
        const std::string &code,
        std::vector<std::vector<Node>> &beams
    ) const
    {
        return decode(code, "", beams, beam_size);
    }

    bool decode(
        const std::string &code,
        const std::string &text,
        std::vector<std::vector<Node>> &beams
    ) const
    {
        return decode(code, text, beams, beam_size);
    }

    bool decode(
        const std::string &code,
        size_t max_path,
        std::vector<std::vector<Node>> &paths,
        std::vector<double> &probs
    ) const;

    std::vector<std::vector<Node>> decode(const std::string &code, size_t max_path = 10) const
    {
        std::vector<std::vector<Node>> beams;
        decode(code, beams);
        return get_paths(beams, max_path);
    }

    std::ostream & output_paths(
        std::ostream &os,
        const std::string &code,
        const std::vector<std::vector<Node>> &paths
    ) const;

    bool update(
        const std::string &code,
        const std::string &text,
        int &index,
        double &prob
    );

    bool update(
        const std::vector<std::string> &codes,
        const std::vector<std::string> &texts,
        std::vector<int> &indeces,
        std::vector<double> &probs
    );

    bool update(
        const std::vector<std::string> &codes,
        const std::vector<std::string> &texts,
        size_t &success,
        size_t &precision,
        double &loss,
        size_t &early_update_count
    );

    std::vector<std::string> predict(const std::string &code, size_t num = 1) const
    {
        auto paths = decode(code, num);
        return get_texts(paths);
    }

    bool predict(
        const std::string &code,
        size_t num,
        std::vector<std::string> &texts,
        std::vector<double> &probs
    ) const
    {
        std::vector<std::vector<Node>> paths;
        if (decode(code, num, paths, probs))
        {
            texts = get_texts(paths);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool predict(
        const std::string &code,
        std::vector<std::string> &texts,
        std::vector<double> &probs
    ) const
    {
        return predict(code, beam_size, texts, probs);
    }

    bool train(std::istream &is, std::map<std::string, double> &metrics);

    /**
     * 训练模型，批量更新版本.
     */
    bool train(
        std::istream &is,
        size_t batch_size,
        std::map<std::string, double> &metrics
    );

    bool train(
        const std::string &fname,
        std::map<std::string, double> &metrics,
        size_t batch_size = 1
    )
    {
        std::ifstream is(fname);
        if (batch_size == 1)
        {
            return train(is, metrics);
        }
        else
        {
            return train(is, batch_size, metrics);
        }
    }

    bool evaluate(
        std::istream &is,
        std::map<std::string, double> &metrics
    ) const;

    bool evaluate(
        const std::string &fname,
        std::map<std::string, double> &metrics
    ) const
    {
        std::ifstream is(fname);
        return evaluate(is, metrics);
    }

    bool save(const std::string &fname) const
    {
        return model.save(fname);
    }

    bool load(const std::string &fname)
    {
        return model.load(fname);
    }

private:
    void init_beams(std::vector<std::vector<Node>> &beams, size_t len) const
    {
        beams.clear();
        beams.reserve(len + 1);
        beams.emplace_back();
        beams.front().emplace_back();
    }

    bool advance(
        const std::string &code,
        const std::string &text,
        size_t pos,
        size_t beam_size,
        std::vector<std::vector<Node>> &beams
    ) const;

    void make_features(
        Node &node,
        const std::string &code,
        size_t pos
    ) const;

    std::vector<std::vector<Node>> get_paths(
        const std::vector<std::vector<Node>> &beams,
        const std::vector<size_t> &indeces
    ) const;

    std::vector<std::vector<Node>> get_paths(
        const std::vector<std::vector<Node>> &beams,
        size_t max_path
    ) const
    {
        std::vector<size_t> indeces;
        for (size_t i = 0; i < std::min(max_path, beams.back().size()); ++i)
        {
            indeces.push_back(i);
        }
        return get_paths(beams, indeces);
    }

    std::vector<std::vector<Node>> get_paths(
        const std::vector<std::vector<Node>> &beams
    ) const
    {
        return get_paths(beams, beams.back().size());
    }

    std::vector<std::string> get_texts(
        const std::vector<std::vector<Node>> &paths
    ) const
    {
        std::vector<std::string> texts;
        texts.reserve(paths.size());

        for (auto &path : paths)
        {
            std::stringstream ss;
            for (auto &node : path)
            {
                if (node.word != nullptr)
                {
                    ss << node.word->text;
                }
            }
            texts.push_back(ss.str());
        }

        return texts;
    }

    /**
     * 使用提早更新（early update）策略计算最优路径.
     *
     * 和经典的提早更新不同之处在于，正确的目标路径可能不止一条，
     * 因此需要当所有目标路径都掉出搜索候选以外才中止
     */
    size_t early_update(
        const std::string &code,
        const std::vector<std::vector<Node>> &dest_beams,
        std::vector<std::vector<Node>> &beams
    ) const;

    int early_update(
        const std::string &code,
        const std::string &text,
        std::vector<std::vector<Node>> &beams,
        std::vector<double> &deltas,
        double &prob
    ) const;

    size_t beam_size;
    const Dictionary &dict;
    Model model;
};

}   // namespace ime

#endif  // _IME_H_
