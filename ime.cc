/**
 * 基于结构化感知机（structured perceptron）的输入法引擎.
 */

#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <utility>
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

    INFO << data.size() << " words loaded" << std::endl;
    return true;
}

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
        weights.emplace(feature, weight);
    }

    INFO << weights.size() << " features loaded" << std::endl;
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
            INFO << "cannot decode code = " << code << ", text = " << text << std::endl;
            return false;
        }
    }

    if (LOG_LEVEL == LOG_DEBUG)
    {
        auto paths = get_paths(beams);
        output_paths(std::cerr, code, paths);
    }

    return true;
}

bool Decoder::decode(
    const std::string &code,
    size_t max_path,
    std::vector<std::vector<Node>> &paths,
    std::vector<double> &probs
) const
{
    std::vector<std::vector<Node>> beams;
    if (decode(code, beams))
    {
        assert(!beams.empty());
        assert(!beams.back().empty());

        double sum = 0;
        for (auto &node : beams.back())
        {
            sum += exp(node.score);
        }

        paths = get_paths(beams, max_path);
        probs.clear();
        probs.reserve(paths.size());
        for (auto & path : paths)
        {
            assert(!path.empty());
            probs.push_back(exp(path.back().score) / sum);
        }

        return true;
    }

    return false;
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
    beam.reserve(beam_size);

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
        node.score = model.score(node.features.begin(), node.features.end());

        // 根据编码子串从词典查找匹配的词进行归约
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
                node.score = model.score(node.features.begin(), node.features.end());
            }
        }
    }

    std::sort(beam.begin(), beam.end(), std::greater<Node>());
    if (beam.size() > beam_size)
    {
        beam.resize(beam_size);
    }

    VERBOSE << "pos = " << pos << std::endl;
    if (LOG_LEVEL <= LOG_VERBOSE)
    {
        auto paths = get_paths(beams);
        output_paths(std::cerr, code, paths);
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

        os << code.substr(rear.code_pos, paths[i].size() - 1 - rear.code_pos) << ' ';

        for (auto &i : rear.features)
        {
            os << i.first << ':' << i.second << ',';
        }
        os << std::endl;
    }
}

bool Decoder::train(std::istream &is, std::map<std::string, double> &metrics)
{
    size_t count = 0;
    size_t prec = 0;
    double loss = 0;

    while (!is.eof())
    {
        std::string line;
        std::string code;
        std::string text;

        std::getline(is, line);
        std::stringstream ss(line);
        ss >> code >> text;

        if (!code.empty() && !text.empty())
        {
            DEBUG << "train sample code = " << code << ", text = " << text << std::endl;

            int index;
            double prob;
            if (update(code, text, index, prob))
            {
                ++count;
                if (index == 0)
                {
                    ++prec;
                }
                loss += -log(prob);

                if (count % 1000 == 0)
                {
                    INFO << count
                        << ": precesion = " << static_cast<double>(prec) / count
                        << ", loss = " << loss / count << std::endl;
                }
            }
        }
    }

    metrics.emplace("precision", static_cast<double>(prec) / count);
    metrics.emplace("loss", loss / count);
    return true;
}

bool Decoder::train(
    std::istream &is,
    size_t batch_size,
    std::map<std::string, double> &metrics
)
{
    size_t batch = 0;
    double precision = 0;
    double loss = 0;
    std::vector<std::string> codes;
    std::vector<std::string> texts;

    while (!is.eof())
    {
        std::string line;
        std::string code;
        std::string text;

        std::getline(is, line);
        std::stringstream ss(line);
        ss >> code >> text;

        if (!code.empty() && !text.empty())
        {
            DEBUG << "train sample code = " << code << ", text = " << text << std::endl;

            codes.push_back(std::move(code));
            texts.push_back(std::move(text));

            if (codes.size() >= batch_size)
            {
                assert(codes.size() == texts.size());

                double prec;
                double l;
                if (update(codes, texts, prec, l))
                {
                    ++batch;
                    precision += prec;
                    loss += l;

                    if (batch % 100 == 0)
                    {
                        INFO << batch
                            << ": precision = " << precision / batch
                            << ", loss = " << loss / batch << std::endl;
                    }
                }

                codes.clear();
                texts.clear();
            }
        }
    }

    if (!codes.empty())
    {
        assert(codes.size() == texts.size());

        double prec;
        double l;
        if (update(codes, texts, prec, l))
        {
            ++batch;
            precision += prec;
            loss += l;
            INFO << batch
                << ": precision = " << precision / batch
                << ", loss = " << loss / batch << std::endl;
        }
    }

    metrics.emplace("precision", precision / batch);
    metrics.emplace("loss", loss / batch);
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
            DEBUG << "early update label = " << beam_size << std::endl;
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

int Decoder::early_update(
    const std::string &code,
    const std::string &text,
    std::vector<std::map<std::string, double>> &features,
    std::vector<double> &deltas,
    double &prob
) const
{
    std::vector<std::vector<Node>> dest_beams;
    decode(code, text, dest_beams);

    // 不是所有路径都能匹配完整汉字串，剔除其中不匹配的
    std::vector<size_t> indeces;
    for (size_t i = 0; i < dest_beams.back().size(); ++i)
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
        for (size_t i = 0; i < dest_beams.back().size(); ++i)
        {
            if (dest_beams.back()[i].text_pos == text.length())
            {
                indeces.push_back(i);
            }
        }

        if (indeces.empty())
        {
            DEBUG << "no match code = " << code << ", text = " << text << std::endl;
            return -1;
        }
    }

    auto paths = get_paths(dest_beams, indeces);
    std::vector<std::vector<Node>> beams;
    auto label = early_update(code, paths, beams);

    // 计算各路径梯度
    double sum = 0;
    for (auto &node : beams.back())
    {
        sum += exp(node.score);
    }

    for (size_t i = 0; i < beams.back().size(); ++i)
    {
        auto &node = beams.back()[i];
        auto p = exp(node.score) / sum;
        auto delta = -p;
        if (i == label)
        {
            prob = p;
            delta += 1;
        }
        features.push_back(std::move(node.features));
        deltas.push_back(delta);
    }

    return label;
}

bool Decoder::update(
    const std::string &code,
    const std::string &text,
    int &index,
    double &prob
)
{
    std::vector<std::map<std::string, double>> features;
    std::vector<double> deltas;
    index = early_update(code, text, features, deltas, prob);
    if (index >= 0)
    {
        assert(features.size() == deltas.size());
        model.update(features, deltas);
        return true;
    }
    else
    {
        INFO << "cannot decode code = " << code << ", text = " << text << std::endl;
        return false;
    }
}

bool Decoder::update(
    const std::vector<std::string> &codes,
    const std::vector<std::string> &texts,
    std::vector<int> &indeces,
    std::vector<double> &probs
)
{
    assert(codes.size() == texts.size());

    auto batch_size = codes.size();
    std::vector<std::vector<std::map<std::string, double>>> batch_features(batch_size);
    std::vector<std::vector<double>> batch_deltas(batch_size);
    indeces.resize(batch_size);
    probs.resize(batch_size);

#pragma omp parallel for num_threads(8)
    // 并行计算梯度
    for (size_t i = 0; i < batch_size; ++i)
    {
        indeces[i] = early_update(
            codes[i],
            texts[i],
            batch_features[i],
            batch_deltas[i],
            probs[i]
        );
    }

    // 批量更新模型
    for (size_t i = 0; i < batch_size; ++i)
    {
        if (indeces[i] >= 0)
        {
            model.update(batch_features[i], batch_deltas[i]);
        }
    }

    return true;
}

bool Decoder::update(
    const std::vector<std::string> &codes,
    const std::vector<std::string> &texts,
    double &precision,
    double &loss
)
{
    std::vector<int> indeces;
    std::vector<double> probs;
    update(codes, texts, indeces, probs);

    precision = 0;
    loss = 0;
    size_t count = 0;
    size_t prec = 0;
    for (size_t i = 0; i < codes.size(); ++i)
    {
        if (indeces[i] >= 0)
        {
            ++count;
            if (indeces[i] == 0)
            {
                ++prec;
            }
            loss -= log(probs[i]);
        }
    }

    precision = static_cast<double>(prec) / count;
    loss /= count;
    return true;
}

bool Decoder::evaluate(
    std::istream &is,
    std::map<std::string, double> &metrics
) const
{
    size_t count = 0;
    size_t prec = 0;
    double loss = 0;

    while (!is.eof())
    {
        std::string line;
        std::string code;
        std::string text;

        std::getline(is, line);
        std::stringstream ss(line);
        ss >> code >> text;

        if (!code.empty() && !text.empty())
        {
            DEBUG << "evaluation sample code = " << code << ", text = " << text << std::endl;

            std::vector<std::string> texts;
            std::vector<double> probs;
            if (predict(code, texts, probs))
            {
                assert(!texts.empty());
                assert(!probs.empty());
                assert(texts.size() == probs.size());

                ++count;
                if (texts.front() == text)
                {
                    ++prec;
                }

                size_t i = 0;
                while ((texts[i] != text) && (i < texts.size()))
                {
                    ++i;
                }

                if (i < texts.size())
                {
                    loss -= log(probs[i]);
                }
                else
                {
                    std::vector<std::vector<Node>> beams;
                    decode(code, beams);
                    assert(!beams.empty());
                    assert(!beams.back().empty());

                    double sum = 0;
                    for (auto &node : beams.back())
                    {
                        sum += exp(node.score);
                    }

                    beams.clear();
                    if (decode(code, text, beams))
                    {
                        assert(!beams.empty());
                        assert(!beams.back().empty());
                        sum += exp(beams.back().front().score);
                        loss += log(sum) - beams.back().front().score;
                    }
                }
            }
        }
    }

    metrics.emplace("precision", static_cast<double>(prec) / count);
    metrics.emplace("loss", loss / count);
    return true;
}

}   // namespace ime
