/**
 *
 */

#include <string>
#include <utility>
#include <iostream>
#include <sstream>

#include "dict.h"
#include "log.h"
#include "common.h"


namespace ime
{

bool Dictionary::load(std::istream &is)
{
    data.clear();
    _max_id = min_id;
    _max_code_len = 0;
    _max_text_len = 0;

    while (!is.eof())
    {
        std::string line;
        std::string code;
        std::string text;

        std::getline(is, line);
        std::stringstream ss(line);
        ss >> code >> text;
        // 丢弃编码或词长度超过限制的词
        if (
            !code.empty() && !text.empty()
            && (code.length() <= code_len_limit)
            && (text.length() <= text_len_limit)
        )
        {
            Word word(_max_id, code, text);
            VERBOSE << "load word " << word << std::endl;
            data.emplace(code, std::move(word));

            ++_max_id;
            if (code.length() > _max_code_len)
            {
                _max_code_len = code.length();
            }
            if (text.length() > _max_text_len)
            {
                _max_text_len = text.length();
            }
        }
        else
        {
            INFO << "drop word " << text << '(' << code << ')' << std::endl;
        }
    }

    INFO << "loaded " << data.size()
        << " words, ID range = [" << min_id << ", " << _max_id
        << "), max code length = " << _max_code_len
        << ", max text length = " << _max_text_len << std::endl;
    return true;
}

}   // namespace ime
