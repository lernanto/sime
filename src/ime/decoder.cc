/**
 * 基于结构化感知机（structured perceptron）的输入法引擎.
 */

#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>

#include "decoder.h"
#include "log.h"
#include "dict.h"


namespace ime
{

std::allocator<Node> Lattice::allocator;

void Lattice::topk(const Node &node)
{
    if (heap.size() == beam_size)
    {
        std::make_heap(heap.begin(), heap.end(), less);
    }
    else if (heap.size() > beam_size)
    {
        std::push_heap(heap.begin(), heap.end(), less);
        std::pop_heap(heap.begin(), heap.end(), less);
    }
}

bool Decoder::decode(
    const std::string &code,
    const std::string &text,
    Lattice &lattice,
    size_t beam_size
) const
{
    DEBUG << "decode code = " << code << ", text = " << text << std::endl;

    lattice.init(code.length(), beam_size);
    for (size_t pos = 1; pos <= code.length(); ++pos)
    {
        if (!advance(code, text, pos, beam_size, lattice))
        {
            INFO << "cannot decode code = " << code << ", text = " << text << std::endl;
            return false;
        }
    }

    DEBUG << lattice << std::endl;
    return true;
}

bool Decoder::advance(
    const std::string &code,
    const std::string &text,
    size_t pos,
    size_t beam_size,
    Lattice &lattice
) const
{
    auto prev_beam = lattice.back();
    assert(!prev_beam.empty());
    lattice.begin_step();

    for (auto &prev_node : prev_beam)
    {
        // 移进
        auto &node = lattice.emplace();
        node.prev = &prev_node;
        node.code_pos = prev_node.code_pos;
        node.text_pos = prev_node.text_pos;
        node.prev_word = (prev_node.word != nullptr) ? &prev_node : prev_node.prev_word;
        make_features(node, code, pos);
        model.compute_score(node);
        lattice.topk(node);

        // 根据编码子串从词典查找匹配的词进行归约
        auto subcode = code.substr(prev_node.code_pos, pos - prev_node.code_pos);
        VERBOSE << "code = " << subcode << std::endl;
        std::multimap<std::string, Word>::const_iterator begin;
        std::multimap<std::string, Word>::const_iterator end;
        dict.find(subcode, begin, end);
        for (auto j = begin; j != end; ++j)
        {
            auto &word = j->second;
            assert(!word.text.empty());

            // 指定是汉字串的情况下，不但要匹配编码串，还要匹配汉字串
            if (text.empty()
                || text.compare(prev_node.text_pos, word.text.length(), word.text) == 0)
            {
                VERBOSE << "code = " << j->first << ", word = " << word.text << std::endl;

                auto &node = lattice.emplace();
                node.prev = &prev_node;
                node.code_pos = pos;
                node.text_pos = prev_node.text_pos + word.text.length();
                node.code = j->first;
                node.word = &word;
                node.prev_word = (prev_node.word != nullptr) ? &prev_node : prev_node.prev_word;
                make_features(node, code, pos);
                model.compute_score(node);
                lattice.topk(node);
            }
        }
    }

    lattice.end_step();

    VERBOSE << "pos = " << pos << std::endl;
    VERBOSE << lattice << std::endl;
    return true;
}

void Decoder::make_features(
    Node &node,
    const std::string &code,
    size_t pos
) const
{
    if (node.word != nullptr)
    {
        node.local_features.push_back(std::make_pair("unigram:" + node.word->text, 1));

        if (node.prev_word != nullptr)
        {
            // 回溯前一个词，构造 bigram
            assert(node.prev_word->word != nullptr);
            assert(!node.prev_word->word->text.empty());
            node.local_features.push_back(std::make_pair(
                "bigram:" + node.prev_word->word->text + "_" + node.word->text,
                1
            ));
        }
    }

    // 当前未匹配编码长度
    if (node.code_pos < pos)
    {
        std::stringstream ss;
        ss << "code_len:" << pos - node.code_pos;
        node.global_features.push_back(std::make_pair(ss.str(), 1));
    }
}

std::ostream & Decoder::output_paths(
    std::ostream &os,
    const std::string &code,
    size_t pos,
    const std::vector<Lattice::ReversePath> &paths
) const {
    for (size_t i = 0; i < paths.size(); ++i)
    {
        os << '#' << i << ": " << paths[i].score() << ' ';

        std::vector<const Node *> path;
        path.reserve(beam_size);
        for (auto j = paths[i].crbegin(); j != paths[i].crend(); ++j)
        {
            path.push_back(&*j);
        }
        for (auto j = path.crbegin(); j != path.crend(); ++j)
        {
            if ((*j)->word != nullptr)
            {
                os << *(*j)->word << ' ';
            }
        }

        auto & rear = paths[i].back();
        os << code.substr(rear.code_pos, pos - rear.code_pos) << ' ';

        model.output_score(os, rear);
        os << std::endl;
    }

    return os;
}

bool Decoder::train(std::istream &is, Metrics &metrics)
{
    size_t count = 0;
    size_t succ = 0;
    size_t prec = 0;
    double loss = 0;
    size_t eu = 0;

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
                ++succ;
                if (index >= beam_size)
                {
                    ++eu;
                }
                if (index == 0)
                {
                    ++prec;
                }
                loss += -log(prob);
            }

            ++count;
            if (count % 1000 == 0)
            {
                INFO << count
                    <<": success rate = " << static_cast<double>(succ) / count
                    << ", precesion = " << static_cast<double>(prec) / succ
                    << ", loss = " << loss / succ
                    << ", early update rate = " << static_cast<double>(eu) / succ << std::endl;
            }
        }
    }

    double success = static_cast<double>(succ) / count;
    double precision = static_cast<double>(prec) / succ;
    loss /= succ;
    double early_update_rate = static_cast<double>(eu) / succ;

    INFO << "count = " << count
        << ", success rate = " << success
        << ", precision = " << precision
        << ", loss = " << loss
        << ", early update rate = " << early_update_rate << std::endl;

    metrics.set("count", count);
    metrics.set("success rate", success);
    metrics.set("precision", precision);
    metrics.set("loss", loss);
    metrics.set("early update rate", early_update_rate);

    return true;
}

bool Decoder::train(std::istream &is, size_t batch_size, Metrics &metrics)
{
    size_t batch = 0;
    size_t count = 0;
    size_t succ = 0;
    size_t prec = 0;
    double loss = 0;
    size_t eu = 0;
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

                if (update(codes, texts, succ, prec, loss, eu))
                {
                    ++batch;
                    count += codes.size();
                    if (batch % 100 == 0)
                    {
                        INFO << batch
                            << ": success rate = " << static_cast<double>(succ) / count
                            << ", precision = " << static_cast<double>(prec) / succ
                            << ", loss = " << loss / succ
                            << ", early update rate = " << static_cast<double>(eu) / succ << std::endl;
                    }

                    codes.clear();
                    texts.clear();
                }
            }
        }
    }

    if (!codes.empty())
    {
        assert(codes.size() == texts.size());

        if (update(codes, texts, succ, prec, loss, eu))
        {
            ++batch;
            count += codes.size();
        }
    }

    double success = static_cast<double>(succ) / count;
    double precision = static_cast<double>(prec) / succ;
    loss /= succ;
    double early_update_rate = static_cast<double>(eu) / succ;

    INFO << "count = " << count
        << ", success rate = " << success
        << ", precision = " << precision
        << ", loss = " << loss
        << ", early update rate = " << early_update_rate << std::endl;

    metrics.set("count", count);
    metrics.set("success rate", success);
    metrics.set("precision", precision);
    metrics.set("loss", loss);
    metrics.set("early update rate", early_update_rate);
    return true;
}

size_t Decoder::early_update(
    const std::string &code,
    const std::vector<std::vector<const Node *>> &paths,
    Lattice &lattice
) const
{
    assert(!paths.empty());
    assert(paths.front().size() == code.length() + 1);

    lattice.init(code.length(), beam_size);

    // 为目标路径初始化祖先节点的索引，用于对比路径
    std::vector<size_t> prev_indeces(paths.size(), 0);
    std::vector<size_t> indeces(paths.size(), beam_size);

    for (size_t pos = 1; pos <= code.length(); ++pos)
    {
        advance(code, "", pos, beam_size, lattice);

        bool found = false;
        for (size_t i = 0; i < paths.size(); ++i)
        {
            indeces[i] = beam_size;
            if (prev_indeces[i] != beam_size)
            {
                for (size_t j = 0; j < lattice[pos].size(); ++j)
                {
                    if ((lattice[pos][j].prev == &lattice[pos - 1][prev_indeces[i]])
                        && (lattice[pos][j].word == paths[i][pos]->word))
                    {
                        indeces[i] = j;
                        found = true;
                        break;
                    }
                }
            }
        }

        if (!found)
        {
            // 目标路径全部掉出搜索范围，提前返回结果
            // 查找祖先节点还在搜索范围内的第一条路径
            size_t i = 0;
            while ((prev_indeces[i] == beam_size) && (i < paths.size()))
            {
                ++i;
            }
            assert(i < paths.size());

            // 把找到的一条目标路径强制加入网格，以便后续计算梯度
            auto &node = lattice.emplace(*paths[i][pos]);
            node.prev = &lattice[pos - 1][prev_indeces[i]];
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

    // 不应到达这里
    assert(false);
    return 0;
}

int Decoder::early_update(
    const std::string &code,
    const std::string &text,
    Lattice &lattice,
    std::vector<double> &deltas,
    double &prob
) const
{
    Lattice dest;
    decode(code, text, dest);

    // 不是所有路径都能匹配完整汉字串，剔除其中不匹配的
    std::vector<size_t> indeces;
    for (size_t i = 0; i < dest.back().size(); ++i)
    {
        if (dest.back()[i].text_pos == text.length())
        {
            indeces.push_back(i);
        }
    }

    if (indeces.empty())
    {
        // 没有搜索到匹配的路径，增加集束大小再试一次
        decode(code, text, dest, beam_size * 2);

        indeces.clear();
        for (size_t i = 0; i < dest.back().size(); ++i)
        {
            if (dest.back()[i].text_pos == text.length())
            {
                indeces.push_back(i);
            }
        }

        if (indeces.empty())
        {
            DEBUG << "cannot decode code = " << code << ", text = " << text << std::endl;
            return -1;
        }
    }

    std::vector<std::vector<const Node *>> paths;
    paths.reserve(beam_size);
    dest.get_paths(indeces.crbegin(), indeces.crend(), std::back_inserter(paths));
    auto label = early_update(code, paths, lattice);

    DEBUG << lattice << std::endl;

    // 计算各路径梯度
    double sum = 0;
    for (auto &node : lattice.back())
    {
        sum += exp(node.score);
    }

    for (size_t i = 0; i < lattice.back().size(); ++i)
    {
        auto &node = lattice.back()[i];
        auto p = exp(node.score) / sum;
        auto delta = -p;
        if (i == label)
        {
            prob = p;
            delta += 1;
        }
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
    Lattice lattice;
    std::vector<double> deltas;
    index = early_update(code, text, lattice, deltas, prob);
    if (index >= 0)
    {
        std::vector<Features> features;
        features.reserve(beam_size);
        for (auto &node : lattice.back())
        {
            features.emplace_back(&node);
        }

        model.update(features, deltas);
        return true;
    }
    else
    {
        DEBUG << "cannot decode code = " << code << ", text = " << text << std::endl;
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
    std::vector<Lattice> batch_lattice(batch_size);
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
            batch_lattice[i],
            batch_deltas[i],
            probs[i]
        );
    }

    // 批量更新模型
    for (size_t i = 0; i < batch_size; ++i)
    {
        if (indeces[i] >= 0)
        {
            std::vector<Features> features;
            features.reserve(beam_size);
            for (auto &node : batch_lattice[i].back())
            {
                features.emplace_back(&node);
            }

            model.update(features, batch_deltas[i]);
        }
    }

    return true;
}

bool Decoder::update(
    const std::vector<std::string> &codes,
    const std::vector<std::string> &texts,
    size_t &success,
    size_t &precision,
    double &loss,
    size_t &early_update_count
)
{
    std::vector<int> indeces;
    std::vector<double> probs;
    update(codes, texts, indeces, probs);

    for (size_t i = 0; i < codes.size(); ++i)
    {
        if (indeces[i] >= 0)
        {
            ++success;
            if (indeces[i] == 0)
            {
                ++precision;
            }
            else if (indeces[i] >= beam_size)
            {
                ++early_update_count;
            }
            loss -= log(probs[i]);
        }
    }

    return true;
}

int Decoder::predict(
    const std::string &code,
    const std::string &text,
    double &prob
) const
{
    int index = -1;
    std::vector<std::string> texts;
    std::vector<double> probs;
    if (predict(code, texts, probs))
    {
        assert(!texts.empty());
        assert(!probs.empty());
        assert(texts.size() == probs.size());

        for (index = 0; (index < texts.size()) && (texts[index] != text); ++index);
        if (index < texts.size())
        {
            prob = probs[index];
        }
        else
        {
            DEBUG << "target text not in beam code = " << code << ", text = " << text << std::endl;

            // 预测结果中没有包含目标文本，无法计算概率，限定文本解码以获取目标文本分数
            Lattice lattice;
            decode(code, lattice);
            double sum = 0;
            for (auto &node : lattice.back())
            {
                sum += exp(node.score);
            }

            if (decode(code, text, lattice))
            {
                index = beam_size;
                std::vector<Lattice::ReversePath> paths;
                lattice.get_paths(1, std::back_inserter(paths));
                sum += exp(paths.front().score());
                prob = exp(paths.front().score()) / sum;
            }
        }
    }

    if (index >= 0)
    {
        DEBUG << "predict code = " << code
            << ", text = " << text
            << ", prob = " << prob << std::endl;
    }
    else
    {
        DEBUG << "cannot predict code = " << code
            << ", text = " << text << std::endl;
    }
    return index;
}

bool Decoder::evaluate(std::istream &is, Metrics &metrics) const
{
    size_t count = 0;
    size_t succ = 0;
    size_t prec = 0;
    size_t inbeam = 0;
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

            ++count;
            double prob = 0;
            auto index = predict(code, text, prob);
            if (index >= 0)
            {
                ++succ;
                if (index < beam_size)
                {
                    ++inbeam;
                    if (index == 0)
                    {
                        ++prec;
                    }
                }

                loss -= log(prob);
            }
        }
    }

    metrics.set("count", count);
    metrics.set("success rate", static_cast<double>(succ) / count);
    metrics.set("precision", static_cast<double>(prec) / succ);
    std::stringstream ss;
    ss << "p@" << beam_size;
    metrics.set(ss.str(), static_cast<double>(inbeam) / succ);
    metrics.set("loss", loss / succ);
    return true;
}

bool Decoder::evaluate(std::istream &is, size_t batch_size, Metrics &metrics) const
{
    size_t count = 0;
    size_t succ = 0;
    size_t prec = 0;
    size_t inbeam = 0;
    double loss = 0;

    while (!is.eof())
    {
        std::vector<std::string> codes;
        std::vector<std::string> texts;
        for (size_t i = 0; (i < batch_size) && !is.eof(); ++i)
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
                codes.push_back(std::move(code));
                texts.push_back(std::move(text));
            }
        }

        if (!codes.empty())
        {
            assert(codes.size() == texts.size());
            count += codes.size();

#pragma omp parallel for num_threads(8)
            for (size_t i = 0; i < codes.size(); ++i)
            {
                double prob = 0;
                auto index = predict(codes[i], texts[i], prob);
                if (index >= 0)
                {
                    ++succ;
                    loss -= log(prob);
                    if (index < beam_size)
                    {
                        ++inbeam;
                        if (index == 0)
                        {
                            ++prec;
                        }
                    }
                }
            }
        }
    }

    metrics.set("count", count);
    metrics.set("success rate", static_cast<double>(succ) / count);
    metrics.set("precision", static_cast<double>(prec) / succ);
    std::stringstream ss;
    ss << "p@" << beam_size;
    metrics.set(ss.str(), static_cast<double>(inbeam) / succ);
    metrics.set("loss", loss / succ);
    return true;
}

}   // namespace ime
