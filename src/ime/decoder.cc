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
#include "common.h"
#include "feature.h"
#include "dict.h"


namespace
{

inline bool greater(const ime::Node *a, const ime::Node *b)
{
    return *a > *b;
}

}   // namespace

namespace ime
{

std::allocator<Node> Lattice::allocator;

void Lattice::topk()
{
    VERBOSE << "add node " << *heap.back() << std::endl;

    if (heap.size() == beam_size)
    {
        std::make_heap(heap.begin(), heap.end(), greater);
    }
    else if (heap.size() > beam_size)
    {
        std::push_heap(heap.begin(), heap.end(), greater);
        std::pop_heap(heap.begin(), heap.end(), greater);
        VERBOSE << "drop node " << *heap.back() << std::endl;
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
    auto succ = begin_decode(code, text, beam_size, lattice);

    for (size_t pos = 1; succ && (pos <= code.length()); ++pos)
    {
        succ = advance(code, text, pos, beam_size, lattice);
    }

    if (succ)
    {
        succ = end_decode(code, text, beam_size, lattice);
    }

    if (succ)
    {
        DEBUG << lattice << std::endl;
    }
    else
    {
        INFO << "cannot decode code = " << code << ", text = " << text << std::endl;
    }
    return succ;
}

bool Decoder::begin_decode(
    const std::string &code,
    const std::string &text,
    size_t beam_size,
    Lattice &lattice,
    bool bos
) const
{
    lattice.begin_step();
    auto &node = lattice.emplace();
    if (bos)
    {
        // 添加一个虚拟的句子起始标识，用于构造 n-gram
        node.word = &bos_eos;
    }
    lattice.topk();
    lattice.end_step();

    return true;
}

bool Decoder::end_decode(
    const std::string &code,
    const std::string &text,
    size_t beam_size,
    Lattice &lattice,
    bool eos
) const
{
    // 最后加入一列特殊的节点，以标记归约完全部编码（和文本）的路径
    auto prev_beam = lattice.back();
    lattice.begin_step();

    for (auto &prev_node : prev_beam)
    {
        if ((prev_node.code_pos == code.length())
            && (text.empty() || (prev_node.text_pos == text.length())))
        {
            auto &node = lattice.emplace(&prev_node);
            if (eos)
            {
                // 添加一个虚拟的句子结束标识，用于构造 n-gram
                node.word = &bos_eos;
            }

            make_features(node, code, code.length());
            model.compute_score(node);
            lattice.topk();
        }
    }

    lattice.end_step();
    VERBOSE << "end decode" << std::endl;
    VERBOSE << lattice << std::endl;
    return lattice.back().size() > 0;
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
    assert(prev_beam.size() > 0);
    lattice.begin_step();

    for (auto &prev_node : prev_beam)
    {
        auto subcode = code.substr(prev_node.code_pos, pos - prev_node.code_pos);

        if (fullfill_shift_constraint(prev_node, code, pos))
        {
            VERBOSE << "shift code = " << subcode << std::endl;

            auto &node = lattice.emplace(&prev_node);
            make_features(node, code, pos);
            model.compute_score(node);
            lattice.topk();
        }

        // 根据编码子串从词典查找匹配的词进行归约
        VERBOSE << "code = " << subcode << std::endl;
        std::multimap<std::string, Word>::const_iterator begin;
        std::multimap<std::string, Word>::const_iterator end;
        dict.find(subcode, begin, end);
        for (auto j = begin; j != end; ++j)
        {
            auto &word = j->second;
            assert(!word.text.empty());

            if (fullfill_reduce_constraint(prev_node, code, text, pos, word))
            {
                VERBOSE << "word = " << word << std::endl;

                auto &node = lattice.emplace(
                    &prev_node,
                    pos,
                    prev_node.text_pos + word.text.length(),
                    &word
                );
                make_features(node, code, pos);
                model.compute_score(node);
                lattice.topk();
            }
        }
    }

    lattice.end_step();
    VERBOSE << "pos = " << pos << std::endl;
    VERBOSE << lattice << std::endl;
    return lattice.back().size() > 0;
}

void Decoder::make_features(
    Node &node,
    const std::string &code,
    size_t pos
) const
{
    if (node.word != nullptr)
    {
        if (!node.word->text.empty())
        {
            node.local_features.push_back(std::make_pair("unigram:" + node.word->text, 1));
        }

        if (node.prev_word != nullptr)
        {
            // 回溯前一个词，构造 bigram
            assert(node.prev_word->word != nullptr);
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

        auto &rear = paths[i].back();
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

            size_t index;
            double prob;
            auto pos = update(code, text, index, prob);
            if (pos > 0)
            {
                ++succ;
                if (pos < code.length() + 2)
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
    Lattice &lattice,
    size_t &label
) const
{
    assert(!paths.empty());
    assert(paths.front().size() == code.length() + 2);

    auto succ = true;
    lattice.init(code.length(), beam_size);
    begin_decode(code, "", beam_size, lattice);

    // 为目标路径初始化祖先节点的索引，用于对比路径
    std::vector<size_t> indeces(paths.size(), 0);
    size_t pos;
    for (pos = 1; succ && (pos <= code.length()); ++pos)
    {
        advance(code, "", pos, beam_size, lattice);
        succ = match(lattice, paths, pos, indeces);
    }

    if (succ)
    {
        end_decode(code, "", beam_size, lattice);
        succ = match(lattice, paths, pos, indeces);
    }

    if (succ)
    {
        ++pos;
    }
    else
    {
        DEBUG << "early update pos = " << pos << std::endl;
    }

    // 搜索结果包含至少一条目标路径，返回排在最前的目标路径
    // 由于 match 已经正确设置了 indeces，即使在查找路径失败时仍然有效
    size_t i = 0;
    while ((i < indeces.size()) && (indeces[i] >= lattice.back().size()))
    {
        ++i;
    }
    assert(i < indeces.size());
    label = indeces[i];

    DEBUG << "label = " << label << std::endl;
    DEBUG << lattice << std::endl;
    return pos;
}

size_t Decoder::early_update(
    const std::string &code,
    const std::string &text,
    Lattice &lattice,
    std::vector<double> &deltas,
    size_t &label,
    double &prob
) const
{
    Lattice dest;
    if (!decode(code, text, dest))
    {
        // 没有搜索到匹配的路径，增加集束大小再试一次
        if (!decode(code, text, dest, beam_size * 2))
        {
            DEBUG << "cannot decode code = " << code << ", text = " << text << std::endl;
            return 0;
        }
    }

    std::vector<Lattice::ReversePath> rpaths;
    dest.get_paths(std::back_inserter(rpaths));
    std::vector<std::vector<const Node *>> paths;
    paths.reserve(rpaths.size());
    for (auto &p : rpaths)
    {
        paths.push_back(p.reverse());
    }

    auto pos = early_update(code, paths, lattice, label);
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

    return pos;
}

bool Decoder::match(
    Lattice &lattice,
    const std::vector<std::vector<const Node *>> &paths,
    size_t pos,
    std::vector<size_t> &indeces
) const
{
    assert(!paths.empty());
    assert(pos < lattice.size());
    assert(pos < paths.front().size());
    assert(indeces.size() == paths.size());

    std::vector<size_t> prev_indeces(
        paths.size(),
        std::numeric_limits<size_t>::max()
    );
    prev_indeces.swap(indeces);
    auto found = false;

    for (size_t i = 0; i < paths.size(); ++i)
    {
        if (prev_indeces[i] < lattice[pos - 1].size())
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
        // 目标路径全部掉出集束，查找祖先节点还在集束内的第一条路径
        size_t i = 0;
        while ((i < prev_indeces.size())
            && (prev_indeces[i] >= lattice[pos - 1].size()))
        {
            ++i;
        }
        assert(i < prev_indeces.size());

        // 把找到的一条目标路径强制加入网格，以便后续计算梯度
        auto &node = lattice.emplace(*paths[i][pos]);
        node.prev = &lattice[pos - 1][prev_indeces[i]];
        node.prev_word = (node.prev->word != nullptr) ? node.prev : node.prev_word;

        indeces[i] = lattice[pos].size() - 1;
    }

    return found;
}

size_t Decoder::update(
    const std::string &code,
    const std::string &text,
    size_t &index,
    double &prob
)
{
    Lattice lattice;
    std::vector<double> deltas;
    auto pos = early_update(code, text, lattice, deltas, index, prob);
    if (pos > 0)
    {
        auto rear = lattice.back();
        assert(rear.size() == deltas.size());

        std::vector<Features> features;
        features.reserve(rear.size());
        features.insert(features.end(), rear.begin(), rear.end());

        model.update(
            features.begin(),
            features.end(),
            deltas.begin(),
            deltas.end()
        );
    }
    else
    {
        DEBUG << "cannot decode code = " << code << ", text = " << text << std::endl;
    }

    return pos;
}

void Decoder::update(
    const std::vector<std::string> &codes,
    const std::vector<std::string> &texts,
    std::vector<size_t> &positions,
    std::vector<size_t> &indeces,
    std::vector<double> &probs
)
{
    assert(codes.size() == texts.size());

    auto batch_size = codes.size();
    std::vector<Lattice> batch_lattice(batch_size);
    std::vector<std::vector<double>> batch_deltas(batch_size);
    positions.resize(batch_size);
    indeces.resize(batch_size);
    probs.resize(batch_size);

#pragma omp parallel for
    // 并行计算梯度
    for (size_t i = 0; i < batch_size; ++i)
    {
        positions[i] = early_update(
            codes[i],
            texts[i],
            batch_lattice[i],
            batch_deltas[i],
            indeces[i],
            probs[i]
        );
    }

    // 批量更新模型
    for (size_t i = 0; i < batch_size; ++i)
    {
        if (positions[i] > 0)
        {
            auto rear = batch_lattice[i].back();
            assert(rear.size() == batch_deltas[i].size());

            std::vector<Features> features;
            features.reserve(rear.size());
            features.insert(features.end(), rear.begin(), rear.end());

            model.update(
                features.begin(),
                features.end(),
                batch_deltas[i].begin(),
                batch_deltas[i].end()
            );
        }
    }
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
    std::vector<size_t> positions;
    std::vector<size_t> indeces;
    std::vector<double> probs;
    update(codes, texts, positions, indeces, probs);

    for (size_t i = 0; i < codes.size(); ++i)
    {
        if (positions[i] > 0)
        {
            ++success;
            if (positions[i] < codes[i].length() + 2)
            {
                ++early_update_count;
            }
            if (indeces[i] == 0)
            {
                ++precision;
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

    Lattice lattice;
    if (decode(code, lattice))
    {
        double sum = 0;
        for (auto &node : lattice.back())
        {
            sum += exp(node.score);
        }

        std::vector<Lattice::ReversePath> paths;
        lattice.get_paths(std::back_inserter(paths));
        index = 0;
        while ((index < paths.size()) && (paths[index].text() != text))
        {
            ++index;
        }

        if (index < paths.size())
        {
            prob = exp(lattice.back()[index].score) / sum;
        }
        else
        {
            // 预测结果中没有包含目标文本，无法计算概率，限定文本解码以获取目标文本分数
            DEBUG << "target text not in beam code = " << code
                << ", text = " << text << std::endl;

            Lattice lattice;
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

#pragma omp parallel for
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
