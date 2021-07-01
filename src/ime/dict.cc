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
            Word word(code, text);
            VERBOSE << "load word " << word << std::endl;
            data.emplace(code, std::move(word));
        }
    }

    INFO << data.size() << " words loaded" << std::endl;
    return true;
}

}   // namespace ime
