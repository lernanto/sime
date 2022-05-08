/**
 * 基于结构化感知机（structured perceptron）的输入法引擎.
 */

#ifndef _DECODER_H_
#define _DECODER_H_

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <locale>
#include <codecvt>

#include "log.h"
#include "common.h"
#include "dict.h"
#include "model.h"


namespace ime
{

class Decoder
{
public:
    Decoder(
        const Dictionary &dict_,
        size_t beam_size_ = 20
    ) : dict(dict_), beam_size(beam_size_), model(), bos_eos() {}

    bool decode(
        const CodeString &code,
        const String &text,
        std::vector<std::vector<Node>> &beams,
        size_t beam_size
    ) const;

    bool decode(
        const CodeString &code,
        std::vector<std::vector<Node>> &beams
    ) const
    {
        return decode(code, String(), beams, beam_size);
    }

    bool decode(
        const CodeString &code,
        const String &text,
        std::vector<std::vector<Node>> &beams
    ) const
    {
        return decode(code, text, beams, beam_size);
    }

    bool decode(
        const CodeString &code,
        size_t max_path,
        std::vector<std::vector<Node>> &paths,
        std::vector<double> &probs
    ) const;

    std::vector<std::vector<Node>> decode(const CodeString &code, size_t max_path = 10) const
    {
        std::vector<std::vector<Node>> beams;
        decode(code, beams);
        return get_paths(beams, max_path);
    }

    std::wostream & output_paths(
        std::wostream &os,
        const CodeString &code,
        const std::vector<std::vector<Node>> &paths
    ) const;

    size_t update(
        const CodeString &code,
        const String &text,
        size_t &index,
        double &prob
    );

    void update(
        const std::vector<CodeString> &codes,
        const std::vector<String> &texts,
        std::vector<size_t> &positions,
        std::vector<size_t> &indeces,
        std::vector<double> &probs
    );

    bool update(
        const std::vector<CodeString> &codes,
        const std::vector<String> &texts,
        size_t &success,
        size_t &precision,
        double &loss,
        size_t &early_update_count
    );

    std::vector<String> predict(const CodeString &code, size_t num = 1) const
    {
        auto paths = decode(code, num);
        return get_texts(paths);
    }

    bool predict(
        const CodeString &code,
        size_t num,
        std::vector<String> &texts,
        std::vector<double> &probs
    ) const
    {
        DEBUG << "predict code = " << code << std::endl;

        std::vector<std::vector<Node>> paths;
        if (decode(code, num, paths, probs))
        {
            texts = get_texts(paths);

            for (size_t i = 0; i < texts.size(); ++i)
            {
                DEBUG << '#' << i << ' ' << texts[i] << std::endl;
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    bool predict(
        const CodeString &code,
        std::vector<String> &texts,
        std::vector<double> &probs
    ) const
    {
        return predict(code, beam_size, texts, probs);
    }

    int predict(
        const CodeString &code,
        const String &text,
        double &prob
    ) const;

    bool train(std::wistream &is, Metrics &metrics);

    /**
     * 训练模型，批量更新版本.
     */
    bool train(std::wistream &is, size_t batch_size, Metrics &metrics);

    bool train(const std::string &fname, Metrics &metrics, size_t batch_size = 1)
    {
        std::ifstream is(fname);
        std::wbuffer_convert<std::codecvt_utf8<wchar_t>> conv(is.rdbuf());
        std::wistream wis(&conv);

        if (batch_size == 1)
        {
            return train(wis, metrics);
        }
        else
        {
            return train(wis, batch_size, metrics);
        }
    }

    bool evaluate(std::wistream &is, Metrics &metrics) const;

    bool evaluate(std::wistream &is, size_t batch_size, Metrics &metrics) const;

    bool evaluate(
        const std::string &fname,
        Metrics &metrics,
        size_t batch_size = 1
    ) const
    {
        std::ifstream is(fname);
        std::wbuffer_convert<std::codecvt_utf8<wchar_t>> conv(is.rdbuf());
        std::wistream wis(&conv);

        if (batch_size == 1)
        {
            return evaluate(wis, metrics);
        }
        else
        {
            return evaluate(wis, batch_size, metrics);
        }
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
        beams.reserve(len + 2);
    }

    bool begin_decode(
        const CodeString &code,
        const String &text,
        size_t beam_size,
        std::vector<std::vector<Node>> &beams,
        bool bos = true
    ) const;

    bool end_decode(
        const CodeString &code,
        const String &text,
        size_t beam_size,
        std::vector<std::vector<Node>> &beams,
        bool eos = true
    ) const;

    bool advance(
        const CodeString &code,
        const String &text,
        size_t pos,
        size_t beam_size,
        std::vector<std::vector<Node>> &beams
    ) const;

    /**
     * 移进节点是否满足限制.
     *
     * 为提高转换成功率，避免无效的节点进入集束，制定一些限制规则过滤节点，
     * 只有有可能转换成功的节点才能加入集束
     */
    bool can_shift(
        const Node &prev_node,
        const CodeString &code,
        size_t pos
    ) const
    {
        // 剩余编码长度小于词典最大编码长度才移进，否则后面也不可能检索到词了
        // TODO: 词典中存在词以编码为前缀才归约
        return (pos < code.length())
            && (pos - prev_node.code_pos < dict.max_code_len());
    }

    /**
     * 归约节点是否满足限制.
     */
    bool can_reduce(
        const Node &prev_node,
        const CodeString &code,
        const String &text,
        size_t pos,
        const Word &word
    ) const
    {
        // 指定是汉字串的情况下，不但要匹配编码串，还要匹配汉字串
        // TODO: 后面必须以合法的编码开头才归约
        return text.empty()
            || (text.compare(prev_node.text_pos, word.text.length(), word.text) == 0);
    }

    void make_features(
        Node &node,
        const CodeString &code,
        size_t pos
    ) const;

    void topk(std::vector<Node> &beam, size_t beam_size) const;

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

    std::vector<String> get_texts(
        const std::vector<std::vector<Node>> &paths
    ) const
    {
        std::vector<String> texts;
        texts.reserve(paths.size());

        for (auto &path : paths)
        {
            std::wstringstream ss;
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
        const CodeString &code,
        const std::vector<std::vector<Node>> &paths,
        std::vector<std::vector<Node>> &beams,
        size_t &label
    ) const;

    size_t early_update(
        const CodeString &code,
        const String &text,
        std::vector<std::vector<Node>> &beams,
        std::vector<double> &deltas,
        size_t &label,
        double &prob
    ) const;

    /**
     * 在集束中查找包含目标路径的节点，如果所有路径都不在集束中，向集束强制添加一个路径的节点.
     *
     * indeces 包含了上一步查找匹配到的节点索引，因此不用从头遍历路径，
     * 只需要从上一步匹配的点向后查找就可以了
     */
    bool match(
        std::vector<std::vector<Node>> &beams,
        const std::vector<std::vector<Node>> &paths,
        size_t pos,
        std::vector<size_t> &indeces
    ) const;

    size_t beam_size;
    const Dictionary &dict;
    Model model;
    const Word bos_eos;     ///< 代表句子起始和结束的虚拟词，用于构造 n-gram
};

}   // namespace ime

#endif  // _DECODER_H_
