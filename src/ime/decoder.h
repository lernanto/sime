/**
 * 基于结构化感知机（structured perceptron）的输入法引擎.
 */

#ifndef _DECODER_H_
#define _DECODER_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iterator>
#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#include "log.h"
#include "common.h"
#include "feature.h"
#include "dict.h"
#include "model.h"


namespace ime
{

/**
 * 集束搜索中用于存储搜索结果节点的数据结构.
 */
class Lattice
{
public:
    class ReversePathIterator : public std::iterator<std::input_iterator_tag, Node>
    {
    public:
        explicit ReversePathIterator(const Node *node = nullptr) : p(node) {}

        const Node & operator * () const
        {
            assert(p != nullptr);
            return *p;
        }

        const Node * operator -> () const
        {
            assert(p != nullptr);
            return p;
        }

        bool operator != (const ReversePathIterator &other) const
        {
            return p != other.p;
        }

        ReversePathIterator & operator ++ ()
        {
            assert(p != nullptr);
            p = p->prev;
            return *this;
        }

        ReversePathIterator operator ++ (int)
        {
            assert(p != nullptr);
            auto old = *this;
            p = p->prev;
            return old;
        }

    private:
        const Node *p;
    };

    typedef const Node *const_iterator;

    class Beam
    {
    public:
        Beam(const Node *begin, const Node *end) :
            __begin(begin), __end(end)
        {
            assert(__begin <= __end);
        }

        bool empty() const
        {
            return __begin == __end;
        }

        size_t size() const
        {
            return static_cast<size_t>(__end - __begin);
        }

        const_iterator begin() const
        {
            return const_iterator(__begin);
        }

        const_iterator end() const
        {
            return const_iterator(__end);
        }

        const Node & operator [] (size_t i) const
        {
            return __begin[i];
        }

    private:
        const Node *__begin;
        const Node *__end;
    };

    class ReversePath
    {
    public:
        friend std::ostream & operator << (std::ostream &os, const ReversePath &rpath);

        typedef ReversePathIterator const_reverse_iterator;

        explicit ReversePath(const Node *rear_ = nullptr) : rear(rear_) {}

        ReversePath(const ReversePath &other) : rear(other.rear) {}

        double score() const
        {
            return rear->score;
        }

        std::string text() const
        {
            std::vector<const Node *> path;
            for (auto i = crbegin(); i != crend(); ++i)
            {
                path.push_back(&*i);
            }

            std::stringstream ss;
            for (auto i = path.crbegin(); i != path.crend(); ++i)
            {
                auto &node = *i;
                if (node->word != nullptr)
                {
                    ss << node->word->text;
                }
            }

            return ss.str();
        }

        Features get_features() const
        {
            return Features(rear);
        }

        std::vector<const Node *> reverse() const
        {
            std::vector<const Node *> path;
            for (auto i = crbegin(); i != crend(); ++i)
            {
                path.push_back(&*i);
            }

            std::reverse(path.begin(), path.end());
            return path;
        }

        const Node & back() const
        {
            assert(rear != nullptr);
            return *rear;
        }

        const_reverse_iterator crbegin() const
        {
            return ReversePathIterator(rear);
        }

        const_reverse_iterator crend() const
        {
            return ReversePathIterator(nullptr);
        }

        bool operator > (const ReversePath &other) const
        {
            return *rear > *other.rear;
        }

    private:
        const Node *rear;
    };

public:
    Lattice() :
        length(0),
        beam_size(0),
        capacity(0),
        pool(nullptr),
        limits(),
        heap() {}

    Lattice(size_t len, size_t bs) : Lattice()
    {
        init(len, bs);
    }

    ~Lattice()
    {
        if (pool != nullptr)
        {
            // 析构已使用的节点
            assert(capacity > 0);
            assert(!limits.empty());
            for (auto p = pool; p < limits.back(); ++p)
            {
                std::allocator_traits<std::allocator<Node>>::destroy(allocator, p);
            }

            allocator.deallocate(pool, capacity);
        }
    }

    void init(size_t len, size_t bs)
    {
        length = len;
        beam_size = bs;

        // 多申请一些空间存放1个 BOS、1列 EOS、以及1个临时节点
        auto new_capacity = (len + 1) * bs + 2;
        if (new_capacity > capacity)
        {
            // 需要的空间大于已分配空间，需要重新分配空间
            if (pool != nullptr)
            {
                // 移动数据无法保证数据完整，因此不尝试移动数据到新位置，全部销毁
                // 确保旧空间的节点是连续的，更新过程中会有销毁不完全的问题
                assert(capacity > 0);
                assert(!limits.empty());
                for (auto p = pool; p < limits.back(); ++p)
                {
                    std::allocator_traits<std::allocator<Node>>::destroy(
                        allocator,
                        p
                    );
                }

                allocator.deallocate(pool, capacity);
            }

            capacity = new_capacity;
            assert(capacity > 0);
            pool = allocator.allocate(capacity);
        }

        limits.clear();
        limits.push_back(pool);
        heap.clear();
    }

    void begin_step()
    {
        limits.push_back(limits.back());
        heap.clear();
    }

    void end_step()
    {
        // 如果有临时分配的节点超出了集束的大小，把它移到集束内部的空位
        if (limits.back() > *(limits.cend() - 2) + beam_size)
        {
            --limits.back();
            if (heap.back() != limits.back())
            {
                std::allocator_traits<std::allocator<Node>>::destroy(
                    allocator,
                    heap.back()
                );
                std::allocator_traits<std::allocator<Node>>::construct(
                    allocator,
                    heap.back(),
                    *limits.back()
                );
            }

            std::allocator_traits<std::allocator<Node>>::destroy(
                allocator,
                limits.back()
            );
            heap.pop_back();
        }
    }

    Node * alloc()
    {
        Node *p = nullptr;
        if (limits.back() <= *(limits.cend() - 2) + beam_size)
        {
            // 集束中的数量还没有超出限制，直接分配对象池中下一个空位
            // 注意允许分配最多 beam_size + 1 个节点，其中一个是临时节点
            assert(limits.back() < pool + capacity);
            p = const_cast<Node *>(limits.back()++);
        }
        else
        {
            // 已经分配了限制的节点数，取一个淘汰的节点回收重新分配
            p = const_cast<Node *>(heap.back());
            heap.pop_back();
            std::allocator_traits<std::allocator<Node>>::destroy(allocator, p);
        }
        return p;
    }

    template<typename ... Args>
    Node & emplace(Args&& ... args)
    {
        auto p = alloc();
        std::allocator_traits<std::allocator<Node>>::construct(
            allocator,
            p,
            std::forward<Args>(args) ...
        );
        heap.push_back(p);
        assert(*(limits.cend() - 2) + heap.size() == limits.back());
        return *p;
    }

    void topk();

    Beam get_beam(size_t i) const
    {
        assert(i < limits.size() - 1);
        return Beam(limits[i], limits[i + 1]);
    }

    size_t size() const
    {
        return limits.size() - 1;
    }

    Beam back() const
    {
        assert(limits.size() >= 2);
        return Beam(*(limits.cend() - 2), limits.back());
    }

    Beam operator [] (size_t i) const
    {
        return get_beam(i);
    }

    template<typename Iterator>
    void get_paths(size_t num, Iterator out) const
    {
        std::vector<ReversePath> paths;
        paths.reserve(limits.back() - *(limits.cend() - 2));
        for (auto p = *(limits.cend() - 2); p < limits.back(); ++p)
        {
            paths.emplace_back(p);
        }

        std::sort(paths.begin(), paths.end(), std::greater<ReversePath>());
        if (num < paths.size())
        {
            paths.resize(num);
        }

        for (auto &p : paths)
        {
            *out++ = p;
        }
    }

    template<typename Iterator>
    void get_paths(Iterator out) const
    {
        get_paths(heap.size(), out);
    }

private:
    static std::allocator<Node> allocator;

    size_t length;      ///< 待解码的编码长度
    size_t beam_size;   ///< 集束大小
    size_t capacity;    ///< 申请的对象池大小，单位是对象个数
    Node *pool;         ///< 预分配的节点对象池
    std::vector<const Node *> limits;   ///< 保存每个集束在对象池中的边界
    std::vector<const Node *> heap;     ///< 保存集束搜索过程中分数最高的若干条路径的指针
};

class Decoder
{
public:
    Decoder(
        const Dictionary &dict_,
        size_t beam_size_ = 20
    ) : dict(dict_), beam_size(beam_size_), model(), bos_eos() {}

    bool decode(
        const std::string &code,
        const std::string &text,
        Lattice &lattice,
        size_t beam_size
    ) const;

    bool decode(
        const std::string &code,
        Lattice &lattice
    ) const
    {
        return decode(code, "", lattice, beam_size);
    }

    bool decode(
        const std::string &code,
        const std::string &text,
        Lattice &lattice
    ) const
    {
        return decode(code, text, lattice, beam_size);
    }

    std::ostream & output_paths(
        std::ostream &os,
        const std::string &code,
        size_t pos,
        const std::vector<Lattice::ReversePath> &paths
    ) const;

    size_t update(
        const std::string &code,
        const std::string &text,
        size_t &index,
        double &prob
    );

    void update(
        const std::vector<std::string> &codes,
        const std::vector<std::string> &texts,
        std::vector<size_t> &positions,
        std::vector<size_t> &indeces,
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
        Lattice lattice;
        std::vector<Lattice::ReversePath> paths;
        std::vector<std::string> texts;

        decode(code, lattice);
        paths.reserve(num);
        texts.reserve(num);
        lattice.get_paths(num, std::back_inserter(paths));
        for (auto &p : paths)
        {
            texts.push_back(p.text());
        }

        return texts;
    }

    bool predict(
        const std::string &code,
        size_t num,
        std::vector<std::string> &texts,
        std::vector<double> &probs
    ) const
    {
        DEBUG << "predict code = " << code << std::endl;

        texts.clear();
        probs.clear();
        texts.reserve(num);
        probs.reserve(num);

        Lattice lattice;
        std::vector<Lattice::ReversePath> paths;
        if (decode(code, lattice))
        {
            paths.reserve(num);
            lattice.get_paths(num, std::back_inserter(paths));
            double sum = 0;
            for (auto &p : paths)
            {
                sum += exp(p.score());
            }
            for (size_t i = 0; i < paths.size(); ++i)
            {
                texts.push_back(paths[i].text());
                probs.push_back(exp(paths[i].score()) / sum);
                DEBUG << '#' << i << ' '
                    << probs.back() << ' '
                    << texts.back() << std::endl;
            }
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

    int predict(
        const std::string &code,
        const std::string &text,
        double &prob
    ) const;

    bool train(std::istream &is, Metrics &metrics);

    /**
     * 训练模型，批量更新版本.
     */
    bool train(std::istream &is, size_t batch_size, Metrics &metrics);

    bool train(const std::string &fname, Metrics &metrics, size_t batch_size = 1)
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

    bool evaluate(std::istream &is, Metrics &metrics) const;

    bool evaluate(std::istream &is, size_t batch_size, Metrics &metrics) const;

    bool evaluate(
        const std::string &fname,
        Metrics &metrics,
        size_t batch_size = 1
    ) const
    {
        std::ifstream is(fname);
        if (batch_size == 1)
        {
            return evaluate(is, metrics);
        }
        else
        {
            return evaluate(is, batch_size, metrics);
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
    bool begin_decode(
        const std::string &code,
        const std::string &text,
        size_t beam_size,
        Lattice &lattice,
        bool bos = true
    ) const;

    bool end_decode(
        const std::string &code,
        const std::string &text,
        size_t beam_size,
        Lattice &lattice,
        bool eos = true
    ) const;

    bool advance(
        const std::string &code,
        const std::string &text,
        size_t pos,
        size_t beam_size,
        Lattice &lattice
    ) const;

    /**
     * 移进节点是否满足限制.
     *
     * 为提高转换成功率，避免无效的节点进入集束，制定一些限制规则过滤节点，
     * 只有有可能转换成功的节点才能加入集束
     */
    bool fullfill_shift_constraint(
        const Node &prev_node,
        const std::string &code,
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
    bool fullfill_reduce_constraint(
        const Node &prev_node,
        const std::string &code,
        const std::string &text,
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
        const std::string &code,
        size_t pos
    ) const;

    /**
     * 使用提早更新（early update）策略计算最优路径.
     *
     * 和经典的提早更新不同之处在于，正确的目标路径可能不止一条，
     * 因此需要当所有目标路径都掉出搜索候选以外才中止
     */
    size_t early_update(
        const std::string &code,
        const std::vector<std::vector<const Node *>> &paths,
        Lattice &lattice,
        size_t &label
    ) const;

    size_t early_update(
        const std::string &code,
        const std::string &text,
        Lattice &lattice,
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
        Lattice &lattice,
        const std::vector<std::vector<const Node *>> &paths,
        size_t pos,
        std::vector<size_t> &indeces
    ) const;

    size_t beam_size;
    const Dictionary &dict;
    Model model;
    const Word bos_eos;     ///< 代表句子起始和结束的虚拟词，用于构造 n-gram
};

inline std::ostream & operator << (std::ostream &os, const Lattice::ReversePath &rpath)
{
    std::vector<const Node *> path;
    for (auto i = rpath.crbegin(); i != rpath.crend(); ++i)
    {
        if (i->word != nullptr)
        {
            path.push_back(&*i);
        }
    }
    for (auto i = path.crbegin(); i != path.crend(); ++i)
    {
        auto &node = **i;
        os << *node.word << '(';
        for (auto &f : node.local_features)
        {
            os << f.first << ':' << f.second << ',';
        }
        os << ' ' << node.local_score << ") ";
    }

    if ((rpath.rear != nullptr) && !rpath.rear->global_features.empty())
    {
        os << '(';
        for (auto &f : rpath.rear->global_features)
        {
            os << f.first << ':' << f.second << ',';
        }
        os << ' ' << rpath.rear->score - rpath.rear->local_score << ')';
    }

    return os;
}

inline std::ostream & operator << (std::ostream &os, const Lattice &lattice)
{
    for (size_t i = 0; i < lattice.back().size(); ++i)
    {
        Lattice::ReversePath path(&lattice.back()[i]);
        os << '#' << i << ' ' << path.score() << ' ' << path << std::endl;
    }
    return os;
}

}   // namespace ime

#endif  // _DECODER_H_
