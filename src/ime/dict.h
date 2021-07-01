/**
 * 输入法词典.
 */

#ifndef _DICT_H_
#define _DICT_H_

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
    explicit Dictionary(const std::string &fname) : data()
    {
        load(fname);
    }

    bool load(std::istream &is);

    bool load(const std::string &fname)
    {
        std::ifstream is(fname);
        return load(is);
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
    std::multimap<std::string, Word> data;
};

}   // namespace ime

#endif  // _DICT_H_
