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

#endif  // _DECODER_H_
