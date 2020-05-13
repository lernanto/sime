/**
 * 基于结构化感知机（structured perceptron）的输入法引擎.
 */

#ifndef _IME_H_
#define _IME_H_

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <iostream>
#include <fstream>

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

class Model
{
public:
    explicit Model(double lr = 0.01) : weights(), learning_rate(lr) {}

    double score(std::map<std::string, double> &features) const
    {
        double sum = 0;

        for (auto &i : features) {
            auto iter = weights.find(i.first);
            if (iter != weights.cend())
            {
                sum += i.second * iter->second;
            }
        }

        return sum;
    }

    bool update(const std::map<std::string, double> &features, double delta)
    {
        if (LOG_LEVEL == LOG_DEBUG)
        {
            DEBUG << "update: ";
            for (auto &i : features)
            {
                DEBUG << i.first << ':' << i.second << ',';
            }
            DEBUG << " +" << delta << '*' << learning_rate << std::endl;
        }

        for (auto &i : features)
        {
            DEBUG << i.first << ':' << weights[i.first] << " + " << i.second << '*' << delta << '*' << learning_rate;
            weights[i.first] += i.second * delta * learning_rate;
            DEBUG << " = " << weights[i.first] << std::endl;
        }

        return true;
    }

private:
    std::unordered_map<std::string, double> weights;
    double learning_rate;
};

class Decoder
{
public:
    struct Node
    {
        size_t prev;
        size_t code_pos;
        size_t text_pos;
        std::string code;
        const Word *word;
        std::map<std::string, double> features;
        double score;

        Node() :
            prev(0),
            code_pos(0),
            text_pos(0),
            code(),
            word(nullptr),
            features(),
            score(0) {}

        Node(const Node &other) :
            prev(other.prev),
            code_pos(other.code_pos),
            text_pos(other.text_pos),
            code(other.code),
            word(other.word),
            features(other.features),
            score(other.score) {}

        bool operator > (const Node &other) const
        {
            return score > other.score;
        }
    };

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

    bool update(const std::string &code, const std::string &text);

    std::vector<std::vector<Node>> decode(const std::string &code, size_t max_path = 10) const
    {
        std::vector<std::vector<Node>> beams;
        decode(code, beams);
        return get_paths(beams, max_path);
    }

    std::ostream & output_paths(
        std::ostream &os,
        const std::string &code,
        size_t pos,
        const std::vector<std::vector<Node>> &paths
    ) const;

    std::ostream & output_paths(
        std::ostream &os,
        const std::string &code,
        const std::vector<std::vector<Node>> &paths
    ) const
    {
        return output_paths(os, code, code.length(), paths);
    }

    bool train(std::istream &is);

    bool train(const std::string &fname)
    {
        std::ifstream is(fname);
        return train(is);
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

    std::map<std::string, double> make_features(
        const std::vector<std::vector<Node>> &beams,
        size_t idx
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

    size_t beam_size;
    const Dictionary &dict;
    Model model;
};

}   // namespace ime

#endif  // _IME_H_
