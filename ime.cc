/**
 * 基于结构化感知机（structured perceptron）的输入法引擎.
 */

#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>

#include "ime.h"
#include "log.h"


namespace ime
{

bool Dictionary::load(std::istream &is)
{
    data.clear();

    std::string line;
    std::string code;
    std::string text;

    while (!is.eof())
    {
        std::getline(is, line);
        std::stringstream ss(line);
        ss >> code >> text;
        data.emplace(code, Word(code, text));
    }

    return true;
}

bool Decoder::decode(
    const std::string &code,
    const std::string &text,
    std::vector<std::vector<Node>> &beams,
    size_t beam_size
) const
{
    DEBUG << "decode code = " << code << ", text = " << text << std::endl;

    init_beams(beams, code.length());
    for (size_t pos = 1; pos <= code.length(); ++pos)
    {
        if (!advance(code, text, pos, beam_size, beams))
        {
            return false;
        }
    }

    return true;
}

bool Decoder::advance(
    const std::string &code,
    const std::string &text,
    size_t pos,
    size_t beam_size,
    std::vector<std::vector<Node>> &beams
) const
{
    auto &prev_beam = beams.back();
    beams.emplace_back();
    auto &beam = beams.back();

    for (size_t i = 0; i < prev_beam.size(); ++i)
    {
        auto &prev_node = prev_beam[i];

        // 移进
        beam.emplace_back();
        auto &node = beam.back();
        node.prev = i;
        node.code_pos = prev_node.code_pos;
        node.text_pos = prev_node.text_pos;
        node.features = make_features(beams, beams.back().size() - 1);
        node.score = model.score(node.features);

        // 根据编码子串从词典查找匹配得词进行归约
        auto subcode = code.substr(prev_node.code_pos, pos - prev_node.code_pos);
        VERBOSE << "code = " << subcode << std::endl;
        std::multimap<std::string, Word>::const_iterator begin;
        std::multimap<std::string, Word>::const_iterator end;
        dict.find(subcode, begin, end);
        for (auto j = begin; j != end; ++j)
        {
            auto &word = j->second;

            // 指定是汉字串的情况下，不但要匹配编码串，还要匹配汉字串
            if (text.empty()
                || text.compare(prev_node.text_pos, word.text.length(), word.text) == 0)
            {
                VERBOSE << "code = " << j->first << ", word = " << word.text << std::endl;

                beam.emplace_back();
                auto &node = beam.back();
                node.prev = i;
                node.code_pos = pos;
                node.text_pos = prev_node.text_pos + word.text.length();
                node.code = j->first;
                node.word = &word;
                node.features = make_features(beams, beams.back().size() - 1);
                node.score = model.score(node.features);
            }
        }
    }

    std::sort(beam.begin(), beam.end(), std::greater<Node>());
    if (beam.size() > beam_size)
    {
        beam.resize(beam_size);
    }

    DEBUG << "pos = " << pos << std::endl;
    if (LOG_LEVEL == LOG_DEBUG)
    {
        auto paths = get_paths(beams);
        output_paths(std::cerr, code, pos, paths);
    }

    return true;
}

std::map<std::string, double> Decoder::make_features(
    const std::vector<std::vector<Node>> &beams,
    size_t idx
) const
{
    assert(!beams.empty());
    assert(beams.back().size() > idx);

    auto &node = beams.back()[idx];
    std::map<std::string, double> features;

    if (node.code_pos < beams.size() - 1)
    {
        features.emplace("code_len", beams.size() - 1 - node.code_pos);
    }

    const Word *last_word = nullptr;
    for (auto i = beams.crbegin(); i != beams.crend(); ++i)
    {
        auto &node = (*i)[idx];
        if (node.word)
        {
            features.emplace("unigram:" + node.word->text, 1);
            if (last_word)
            {
                features.emplace("bigram:" + node.word->text + "_" + last_word->text, 1);
            }
            last_word = node.word;
        }
        idx = node.prev;
    }

    return features;
}

std::vector<std::vector<Decoder::Node>> Decoder::get_paths(
    const std::vector<std::vector<Node>> &beams,
    const std::vector<size_t> &indeces
) const {
    assert(!beams.empty());

    std::vector<std::vector<Node>> paths;
    paths.resize(indeces.size());
    std::vector<size_t> idx(indeces);

    for (auto i = beams.crbegin(); i != beams.crend(); ++i)
    {
        for (size_t j = 0; j < paths.size(); ++j)
        {
            auto &node = (*i)[idx[j]];
            paths[j].push_back(node);
            idx[j] = node.prev;
        }
    }

    for (auto &path : paths)
    {
        std::reverse(path.begin(), path.end());
    }

    return paths;
}

std::ostream & Decoder::output_paths(
    std::ostream &os,
    const std::string &code,
    size_t pos,
    const std::vector<std::vector<Node>> &paths
) const {
    for (size_t i = 0; i < paths.size(); ++i)
    {
        assert(!paths[i].empty());

        auto &rear = paths[i].back();

        os << '#' << i << ": " << rear.score << ' ';

        for (auto &node : paths[i])
        {
            if (node.word)
            {
                os << node.word->text << ' ';
            }
        }

        os << code.substr(rear.code_pos, pos - rear.code_pos) << ' ';

        for (auto &i : rear.features)
        {
            os << i.first << ':' << i.second << ',';
        }
        os << std::endl;
    }
}

bool Decoder::train(std::istream &is)
{
    std::string line;
    std::string code;
    std::string text;

    while (!is.eof())
    {
        std::getline(is, line);
        std::stringstream ss(line);
        ss >> code >> text;
        update(code, text);
    }

    return true;
}

size_t Decoder::early_update(
    const std::string &code,
    const std::vector<std::vector<Node>> &paths,
    std::vector<std::vector<Node>> &beams
) const
{
    assert(!paths.empty());
    assert(paths.front().size() == code.length() + 1);

    init_beams(beams, code.length());

    // 为目标路径初始化祖先节点的索引，用于对比路径
    std::vector<size_t> prev_indeces(paths.size(), 0);
    std::vector<size_t> indeces(paths.size(), beam_size);

    for (size_t pos = 1; pos <= code.length(); ++pos)
    {
        advance(code, "", pos, beam_size, beams);

        bool hit = false;
        for (size_t i = 0; i < paths.size(); ++i)
        {
            indeces[i] = beam_size;
            if (prev_indeces[i] != beam_size)
            {
                for (size_t j = 0; j < beams[pos].size(); ++j)
                {
                    if ((prev_indeces[i] == beams[pos][j].prev)
                        && (paths[i][pos].word == beams[pos][j].word))
                    {
                        indeces[i] = j;
                        hit = true;
                        break;
                    }
                }
            }
        }

        if (!hit)
        {
            // 目标路径全部掉出搜索范围，提前返回结果
            // 查找祖先节点还在搜索范围内的第一条路径
            size_t i = 0;
            while ((prev_indeces[i] == beam_size) && (i < paths.size()))
            {
                ++i;
            }
            assert(i < paths.size());

            beams[pos].push_back(paths[i][pos]);
            beams[pos].back().prev = prev_indeces[i];
            DEBUG << "label = " << beam_size << std::endl;
            return beam_size;
        }

        prev_indeces.swap(indeces);
    }

    // 搜索结果包含至少一条目标路径，返回排在最前的目标路径
    for (size_t i = 0; i < prev_indeces.size(); ++i)
    {
        if (prev_indeces[i] != beam_size)
        {
            DEBUG << "label = " << prev_indeces[i] << std::endl;
            return prev_indeces[i];
        }
    }
}

bool Decoder::update(const std::string &code, const std::string &text)
{
    std::vector<std::vector<Node>> dest_beams;
    decode(code, text, dest_beams);

    // 不是所有路径都能匹配完整汉字串，剔除其中不匹配的
    std::vector<size_t> indeces;
    for (size_t i = 0; i < dest_beams.size(); ++i)
    {
        if (dest_beams.back()[i].text_pos == text.length())
        {
            indeces.push_back(i);
        }
    }

    if (indeces.empty())
    {
        // 没有搜索到匹配的路径，增加集束大小再试一次
        decode(code, text, dest_beams, beam_size * 2);

        indeces.clear();
        for (size_t i = 0; i < dest_beams.size(); ++i)
        {
            if (dest_beams.back()[i].text_pos == text.length())
            {
                indeces.push_back(i);
            }
        }

        if (indeces.empty())
        {
            INFO << "cannot decode code = " << code << ", text = " << text << std::endl;
            return false;
        }
    }

    auto paths = get_paths(dest_beams, indeces);
    std::vector<std::vector<Node>> beams;
    auto label = early_update(code, paths, beams);

    // 更新模型
    double sum = 0;
    for (auto &node : beams.back())
    {
        sum += exp(node.score);
    }

    for (size_t i = 0; i < beams.back().size(); ++i)
    {
        auto &node = beams.back()[i];
        auto prob = exp(node.score) / sum;
        auto delta = ((i == label) ? 1 - prob : -prob);
        model.update(node.features, delta);
    }

    return true;
}

}   // namespace ime
