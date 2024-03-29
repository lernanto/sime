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
    explicit Dictionary(
        const std::string &fname,
        size_t code_len_limit_ = std::numeric_limits<size_t>::max(),
        size_t text_len_limit_ = std::numeric_limits<size_t>::max()
    ) :
        code_len_limit(code_len_limit_),
        text_len_limit(text_len_limit_),
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

private:
    size_t code_len_limit;      ///< 最大编码长度限制
    size_t text_len_limit;      ///< 最大词长限制
    size_t _max_code_len;       ///< 实际载入的最大编码长度
    size_t _max_text_len;       ///< 实际载入的最大词长
    std::multimap<std::string, Word> data;
};

}   // namespace ime

#endif  // _DICT_H_
