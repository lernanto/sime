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

#include "decoder.h"
#include "log.h"
#include "dict.h"


namespace ime
{

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

    if (LOG_LEVEL <= LOG_DEBUG)
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

    for (auto & prev_node : prev_beam)
    {
        // 移进
        beam.emplace_back();
        auto &node = beam.back();
        node.prev = &prev_node;
        node.code_pos = prev_node.code_pos;
        node.text_pos = prev_node.text_pos;
        node.prev_word = (prev_node.word != nullptr) ? &prev_node : prev_node.prev_word;
        make_features(node, code, pos);
        model.compute_score(node);

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

                beam.emplace_back();
                auto &node = beam.back();
                node.prev = &prev_node;
                node.code_pos = pos;
                node.text_pos = prev_node.text_pos + word.text.length();
                node.code = j->first;
                node.word = &word;
                node.prev_word = (prev_node.word != nullptr) ? &prev_node : prev_node.prev_word;
                make_features(node, code, pos);
                model.compute_score(node);
            }
        }
    }

    std::vector<const Node *> tosort;
    tosort.reserve(beam.size());
    for (auto &node : beam)
    {
        tosort.push_back(&node);
    }

    std::sort(
        tosort.begin(),
        tosort.end(),
        [](const Node *a, const Node *b) { return *a > *b; }
    );
    if (tosort.size() > beam_size)
    {
        tosort.resize(beam_size);
    }

    std::vector<Node> new_beam;
    new_beam.reserve(tosort.size());
    for (auto &node : tosort)
    {
        new_beam.emplace_back(*node);
    }
    beam.swap(new_beam);

    VERBOSE << "pos = " << pos << std::endl;
    if (LOG_LEVEL <= LOG_VERBOSE)
    {
        auto paths = get_paths(beams);
        output_paths(std::cerr, code, paths);
    }

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

std::vector<std::vector<Node>> Decoder::get_paths(
    const std::vector<std::vector<Node>> &beams,
    const std::vector<size_t> &indeces
) const {
    assert(!beams.empty());

    std::vector<std::vector<Node>> paths;
    paths.reserve(indeces.size());

    for (auto i : indeces)
    {
        paths.emplace_back();
        auto & path = paths.back();

        for (auto p = &beams.back()[i]; p != nullptr; p = p->prev)
        {
            path.push_back(*p);
        }

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

        bool found = false;
        for (size_t i = 0; i < paths.size(); ++i)
        {
            indeces[i] = beam_size;
            if (prev_indeces[i] != beam_size)
            {
                for (size_t j = 0; j < beams[pos].size(); ++j)
                {
                    if ((beams[pos][j].prev == &beams[pos - 1][prev_indeces[i]])
                        && (beams[pos][j].word == paths[i][pos].word))
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

            beams[pos].push_back(paths[i][pos]);
            beams[pos].back().prev = &beams[pos - 1][prev_indeces[i]];
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
    std::vector<std::vector<Node>> &beams,
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
            DEBUG << "cannot decode code = " << code << ", text = " << text << std::endl;
            return -1;
        }
    }

    auto paths = get_paths(dest_beams, indeces);
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
    std::vector<std::vector<Node>> beams;
    std::vector<double> deltas;
    index = early_update(code, text, beams, deltas, prob);
    if (index >= 0)
    {
        assert(beams.back().size() == deltas.size());
        model.update(beams.back(), deltas);
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
    std::vector<std::vector<std::vector<Node>>> batch_beams(batch_size);
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
            batch_beams[i],
            batch_deltas[i],
            probs[i]
        );
    }

    // 批量更新模型
    for (size_t i = 0; i < batch_size; ++i)
    {
        if (indeces[i] >= 0)
        {
            assert(batch_beams[i].back().size() == batch_deltas[i].size());
            model.update(batch_beams[i].back(), batch_deltas[i]);
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
                index = beam_size;
                sum += exp(beams.back().front().score);
                prob = exp(beams.back().front().score) / sum;
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
