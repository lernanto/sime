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
    auto succ = begin_decode(code, text, beam_size, beams);

    for (size_t pos = 1; succ && (pos <= code.length()); ++pos)
    {
        succ = advance(code, text, pos, beam_size, beams);
    }

    if (succ)
    {
        succ = end_decode(code, text, beam_size, beams);
    }

    if (succ)
    {
        if (LOG_LEVEL <= LOG_DEBUG)
        {
            auto paths = get_paths(beams);
            output_paths(std::cerr, code, paths);
        }
        return true;
    }
    else
    {
        INFO << "cannot decode code = " << code << ", text = " << text << std::endl;
        return false;
    }
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

bool Decoder::begin_decode(
    const std::string &code,
    const std::string &text,
    size_t beam_size,
    std::vector<std::vector<Node>> &beams,
    bool bos
) const
{
    beams.emplace_back();
    beams.back().emplace_back();
    if (bos)
    {
        // 添加一个虚拟的句子起始标识，用于构造 n-gram
        beams.back().back().word = &bos_eos;
    }

    return true;
}

bool Decoder::end_decode(
    const std::string &code,
    const std::string &text,
    size_t beam_size,
    std::vector<std::vector<Node>> &beams,
    bool eos
) const
{
    // 最后加入一列特殊的节点，以标记归约完全部编码（和文本）的路径
    auto &prev_beam = beams.back();
    beams.emplace_back();
    auto &beam = beams.back();

    for (auto &prev_node : prev_beam)
    {
        if ((prev_node.code_pos == code.length())
            && (text.empty() || (prev_node.text_pos == text.length())))
        {
            beam.emplace_back(&prev_node);
            auto &node = beam.back();

            if (eos)
            {
                // 添加一个虚拟的句子结束标识，用于构造 n-gram
                node.word = &bos_eos;
            }

            make_features(node, code, code.length());
            model.compute_score(node);
        }
    }

    if (!beam.empty())
    {
        topk(beam, beam_size);

        VERBOSE << "end decode" << std::endl;
        if (LOG_LEVEL <= LOG_VERBOSE)
        {
            auto paths = get_paths(beams);
            output_paths(std::cerr, code, paths);
        }

        return true;
    }
    else
    {
        return false;
    }
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

    for (auto &prev_node : prev_beam)
    {
        auto subcode = code.substr(prev_node.code_pos, pos - prev_node.code_pos);

        if (can_shift(prev_node, code, pos))
        {
            VERBOSE << "shift code = " << subcode << std::endl;

            beam.emplace_back(&prev_node);
            auto &node = beam.back();
            make_features(node, code, pos);
            model.compute_score(node);
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

            if (can_reduce(prev_node, code, text, pos, word))
            {
                VERBOSE << "reduce word = " << word << std::endl;

                beam.emplace_back(&prev_node, pos, prev_node.text_pos + word.text.length(), &word);
                auto &node = beam.back();
                make_features(node, code, pos);
                model.compute_score(node);
            }
        }
    }

    if (!beam.empty())
    {
        topk(beam, beam_size);

        VERBOSE << "pos = " << pos << std::endl;
        if (LOG_LEVEL <= LOG_VERBOSE)
        {
            auto paths = get_paths(beams);
            output_paths(std::cerr, code, paths);
        }

        return true;
    }
    else
    {
        return false;
    }
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
    std::stringstream ss;
    ss << "code_len:" << pos - node.code_pos;
    node.global_features.push_back(std::make_pair(ss.str(), 1));
}

void Decoder::topk(std::vector<Node> &beam, size_t beam_size) const
{
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
    const std::vector<std::vector<Node>> &paths,
    std::vector<std::vector<Node>> &beams,
    size_t &label
) const
{
    assert(!paths.empty());
    assert(paths.front().size() == code.length() + 2);

    auto succ = true;
    init_beams(beams, code.length());
    begin_decode(code, "", beam_size, beams);

    // 为目标路径初始化祖先节点的索引，用于对比路径
    std::vector<size_t> indeces(paths.size(), 0);
    size_t pos;
    for (pos = 1; succ && (pos <= code.length()); ++pos)
    {
        advance(code, "", pos, beam_size, beams);
        succ = match(beams, paths, pos, indeces);
    }

    if (succ)
    {
        end_decode(code, "", beam_size, beams);
        succ = match(beams, paths, pos, indeces);
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
    while ((i < indeces.size()) && (indeces[i] >= beams.back().size()))
    {
        ++i;
    }
    assert(i < indeces.size());
    label = indeces[i];

    DEBUG << "label = " << label << std::endl;
    if (LOG_LEVEL <= LOG_DEBUG)
    {
        auto paths = get_paths(beams);
        output_paths(std::cerr, code, paths);
    }

    return pos;
}

size_t Decoder::early_update(
    const std::string &code,
    const std::string &text,
    std::vector<std::vector<Node>> &beams,
    std::vector<double> &deltas,
    size_t &label,
    double &prob
) const
{
    std::vector<std::vector<Node>> dest_beams;
    if (!decode(code, text, dest_beams))
    {
        // 没有搜索到匹配的路径，增加集束大小再试一次
        if (!decode(code, text, dest_beams, beam_size * 2))
        {
            DEBUG << "cannot decode code = " << code << ", text = " << text << std::endl;
            return 0;
        }
    }

    auto paths = get_paths(dest_beams);
    auto pos = early_update(code, paths, beams, label);

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

    return pos;
}

bool Decoder::match(
    std::vector<std::vector<Node>> &beams,
    const std::vector<std::vector<Node>> &paths,
    size_t pos,
    std::vector<size_t> &indeces
) const
{
    assert(!paths.empty());
    assert(pos < beams.size());
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
        if (prev_indeces[i] < beams[pos - 1].size())
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
        // 目标路径全部掉出集束，查找祖先节点还在集束内的第一条路径
        size_t i = 0;
        while ((i < prev_indeces.size())
            && (prev_indeces[i] >= beams[pos - 1].size()))
        {
            ++i;
        }
        assert(i < prev_indeces.size());

        beams[pos].emplace_back(paths[i][pos]);
        auto &node = beams[pos].back();
        node.prev = &beams[pos - 1][prev_indeces[i]];
        node.prev_word = (node.prev->word != nullptr) ? node.prev : node.prev_word;

        indeces[i] = beams[pos].size() - 1;
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
    std::vector<std::vector<Node>> beams;
    std::vector<double> deltas;
    auto pos = early_update(code, text, beams, deltas, index, prob);
    if (pos > 0)
    {
        assert(beams.back().size() == deltas.size());

        std::vector<Features> features;
        features.reserve(beams.back().size());
        features.insert(
            features.end(),
            beams.back().cbegin(),
            beams.back().cend()
        );

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
    std::vector<std::vector<std::vector<Node>>> batch_beams(batch_size);
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
            batch_beams[i],
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
            auto &rear = batch_beams[i].back();
            assert(rear.size() == batch_deltas[i].size());

            std::vector<Features> features;
            features.reserve(rear.size());
            features.insert(features.end(), rear.cbegin(), rear.cend());

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

    std::vector<std::vector<Node>> beams;
    if (decode(code, beams))
    {
        double sum = 0;
        for (auto &node : beams.back())
        {
            sum += exp(node.score);
        }

        auto texts = get_texts(get_paths(beams));
        for (index = 0; (index < texts.size()) && (texts[index] != text); ++index);
        if (index < texts.size())
        {
            prob = exp(beams.back()[index].score) / sum;
        }
        else
        {
            // 预测结果中没有包含目标文本，无法计算概率，限定文本解码以获取目标文本分数
            DEBUG << "target text not in beam code = " << code
                << ", text = " << text << std::endl;

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
