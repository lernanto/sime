/**
 * 特征相关的数据结构.
 */

#ifndef _FEATURE_H_
#define _FEATURE_H_

#include <iterator>

#include "common.h"


namespace ime
{

class Features
{
public:
    class Iterator : public std::iterator<std::input_iterator_tag, Features>
    {
    public:
        explicit Iterator(const Node *p = nullptr) : node(p), local(false)
        {
            if (node != nullptr)
            {
                _cur = node->global_features.cbegin();
                _end = node->global_features.cend();
                force_valid();
            }
        }

        void force_valid()
        {
            while ((_cur == _end) && (node != nullptr))
            {
                if (local == false)
                {
                    local = true;
                    _cur = node->local_features.cbegin();
                    _end = node->local_features.cend();
                }
                else
                {
                    node = node->prev;
                    if (node != nullptr)
                    {
                        _cur = node->local_features.cbegin();
                        _end = node->local_features.cend();
                    }
                }
            }
        }

        bool operator == (const Iterator &other) const
        {
            return ((node == nullptr) && (other.node == nullptr))
                || ((node == other.node) && (local == other.local) && (_cur == other._cur));
        }

        bool operator != (const Iterator &other) const
        {
            return !(*this == other);
        }

        Iterator & operator ++ ()
        {
            ++_cur;
            force_valid();
            return *this;
        }

        Iterator operator ++ (int)
        {
            auto old = *this;
            _cur++;
            force_valid();
            return old;
        }

        const std::pair<std::string, double> & operator * () const
        {
            return *_cur;
        }

        const std::pair<std::string, double> * operator -> () const
        {
            return &*_cur;
        }

    private:
        const Node *node;
        bool local;
        std::vector<std::pair<std::string, double>>::const_iterator _cur;
        std::vector<std::pair<std::string, double>>::const_iterator _end;
    };

    typedef Iterator const_reverse_iterator;

    explicit Features(const Node *rear_) : rear(rear_) {}

    Features(const Node &rear_) : rear(&rear_) {}

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

inline std::ostream & operator << (std::ostream &os, const Features &features)
{
    for (auto &f : features)
    {
        os << f.first << ':' << f.second <<',';
    }
    return os;
}

}   // namespace ime

#endif  // _FEATURE_H_
