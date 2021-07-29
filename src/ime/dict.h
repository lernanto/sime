/**
 * 输入法词典.
 */

#ifndef _DICT_H_
#define _DICT_H_

#include <limits>
#include <string>
#include <map>
#include <fstream>
#include <iostream>

#include "common.h"


namespace ime
{

class Dictionary
{
public:
    explicit Dictionary(const std::string &fname) :
        Dictionary(
            fname,
            2,
            std::numeric_limits<size_t>::max(),
            std::numeric_limits<size_t>::max()
    ) {}

    Dictionary(const std::string &fname, size_t code_len_limit_) :
        Dictionary(fname, 2, code_len_limit_, std::numeric_limits<size_t>::max()) {}

    Dictionary(
        const std::string &fname,
        unsigned min_id_,
        size_t code_len_limit_,
        size_t text_len_limit_
    ) :
        min_id(min_id_),
        code_len_limit(code_len_limit_),
        text_len_limit(text_len_limit_),
        _max_id(min_id_),
        _max_code_len(0),
        _max_text_len(0),
        data()
    {
        load(fname);
    }

    bool load(std::istream &is);

    bool load(const std::string &fname)
    {
        std::ifstream is(fname);
        return load(is);
    }

    unsigned int max_id() const
    {
        return _max_id;
    }

    size_t max_code_len() const
    {
        return _max_code_len;
    }

    size_t max_text_len() const
    {
        return _max_text_len;
    }

    void find(
        const std::string &code,
        std::multimap<std::string, Word>::const_iterator &begin,
        std::multimap<std::string, Word>::const_iterator &end
    ) const
    {
        auto range = data.equal_range(code);
        begin = range.first;
        end = range.second;
    }

    const unsigned int min_id;     ///< 自动分配的词 ID 从该基数开始分配
    const size_t code_len_limit;    ///< 最大编码长度限制
    const size_t text_len_limit;    ///< 最大词长限制

private:
    unsigned int _max_id;           ///< 实际分配的最大 ID + 1
    size_t _max_code_len;           ///< 实际载入的最大编码长度
    size_t _max_text_len;           ///< 实际载入的最大词长
    std::multimap<std::string, Word> data;
};

}   // namespace ime

#endif  // _DICT_H_
