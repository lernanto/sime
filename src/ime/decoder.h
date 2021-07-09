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
#include "dict.h"
#include "model.h"


namespace ime
{

class Features
{
public:
    class Iterator
    {
    public:
        explicit Iterator(const Node *p = nullptr) : node(p), local(false)
        {
            if (node != nullptr)
            {
                iter = node->global_features.cbegin();
                __end = node->global_features.cend();
                force_valid();
            }
        }

        void force_valid()
        {
            while ((iter == __end) && (node != nullptr))
            {
                if (local == false)
                {
                    local = true;
                    iter = node->local_features.cbegin();
                    __end = node->local_features.cend();
                }
                else
                {
                    node = node->prev;
                    if (node != nullptr)
                    {
                        iter = node->local_features.cbegin();
                        __end = node->local_features.cend();
                    }
                }
            }
        }

        bool operator == (const Iterator &other) const
        {
            return ((node == nullptr) && (other.node == nullptr))
                || ((node == other.node) && (local == other.local) && (iter == other.iter));
        }

        bool operator != (const Iterator &other) const
        {
            return !(*this == other);
        }

        Iterator & operator ++ ()
        {
            ++iter;
            force_valid();
            return *this;
        }

        Iterator operator ++ (int)
        {
            auto old = *this;
            iter++;
            force_valid();
            return old;
        }

        const std::pair<std::string, double> & operator * () const
        {
            return *iter;
        }

        const std::pair<std::string, double> * operator -> () const
        {
            return &*iter;
        }

    private:
        const Node *node;
        bool local;
        std::vector<std::pair<std::string, double>>::const_iterator iter;
        std::vector<std::pair<std::string, double>>::const_iterator __end;
    };

    typedef Iterator const_reverse_iterator;

    Features(const Node *rear_) : rear(rear_) {}

    const_reverse_iterator begin() const
    {
        return Iterator(rear);
    }

    const_reverse_iterator end() const
    {
        return Iterator();
    }

private:
    const Node *rear;
};

/**
 * 集束搜索中用于存储搜索结果节点的数据结构.
 */
class Lattice
{
public:
    class ReversePathIterator : public std::iterator<std::forward_iterator_tag, Node>
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

        bool operator < (const ReversePath &other) const
        {
            return rear->score > other.rear->score;
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

        // 多申请一些空间存放根节点和临时节点
        auto new_capacity = len * bs + 2;
        if (new_capacity > capacity)
        {
            // 需要的空间大于已分配空间，需要重新分配空间
            if (pool != nullptr)
            {
                // 移动数据无法保证数据完整，因此不尝试移动数据到新位置，全部销毁
                // 确保旧空间的节点是连续的，更新过程中会有销毁不完全的问题
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
            pool = allocator.allocate(capacity);
        }

        limits.clear();
        heap.clear();

        assert(pool != nullptr);
        std::allocator_traits<std::allocator<Node>>::construct(allocator, pool);
        limits.push_back(pool);
        limits.push_back(pool + 1);
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

    Node & emplace()
    {
        auto p = alloc();
        std::allocator_traits<std::allocator<Node>>::construct(allocator, p);
        heap.push_back(p);
        assert(*(limits.cend() - 2) + heap.size() == limits.back());
        return *p;
    }

    Node & emplace(const Node &node)
    {
        auto p = alloc();
        std::allocator_traits<std::allocator<Node>>::construct(allocator, p, node);
        heap.push_back(p);
        assert(*(limits.cend() - 2) + heap.size() == limits.back());
        return *p;
    }

    void topk(const Node &node);

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
        paths.reserve(beam_size);
        for (auto p = *(limits.cend() - 2); p < limits.back(); ++p)
        {
            paths.emplace_back(p);
        }

        std::sort(paths.begin(), paths.end());
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

    template<typename InputIterator, typename OutputIterator>
    void get_paths(InputIterator begin, InputIterator end, OutputIterator out) const
    {
        for (auto i = begin; i < end; ++i)
        {
            std::vector<const Node *> path;
            path.reserve(limits.size());
            for (auto p = *(limits.cend() - 2) + *i; p != nullptr; p = p->prev)
            {
                path.push_back(p);
            }
            std::reverse(path.begin(), path.end());
            *out++ = std::move(path);
        }
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
    ) : dict(dict_), beam_size(beam_size_), model() {}

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
    bool advance(
        const std::string &code,
        const std::string &text,
        size_t pos,
        size_t beam_size,
        Lattice &lattice
    ) const;

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
        Lattice &lattice
    ) const;

    int early_update(
        const std::string &code,
        const std::string &text,
        Lattice &lattice,
        std::vector<double> &deltas,
        double &prob
    ) const;

    size_t beam_size;
    const Dictionary &dict;
    Model model;
};

inline std::ostream & operator << (std::ostream &os, const Features &features)
{
    for (auto &f : features)
    {
        os << f.first << ':' << f.second <<',';
    }
    return os;
}

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
