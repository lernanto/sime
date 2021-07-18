/**
 *
 */

#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <iostream>

#include "ime/dict.h"
#include "ime/decoder.h"


int main(int argc, char **argv)
{

    std::string dict_file = argv[1];
    std::string model_file = argv[2];

    ime::Dictionary dict(dict_file, 20);
    ime::Decoder decoder(dict);
    decoder.load(model_file);

    while (!std::cin.eof())
    {
        std::string code;
        std::cin >> code;

        std::vector<std::string> texts;
        std::vector<double> probs;
        if (decoder.predict(code, 10, texts, probs))
        {
            assert(texts.size() == probs.size());

            for (size_t i = 0; i < texts.size(); ++i)
            {
                std::cout << i + 1 << ": " << texts[i] << ' ' << probs[i] << std::endl;
            }
        }
    }

    return 0;
}
