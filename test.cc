/**
 *
 */

#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <iostream>

#include "ime.h"


int main(int argc, char **argv)
{
    ime::Dictionary dict("test.dic");

    ime::Decoder decoder(dict);
    decoder.train("test.txt");

    const std::string code = "ceshiceshi";
    auto paths = decoder.decode(code);
    std::cout << "paths: " << std::endl;
    decoder.output_paths(std::cout, code, paths);

    return 0;
}
