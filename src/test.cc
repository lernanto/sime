/**
 *
 */

#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <iostream>

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else   // _WIN32
#include <sys/io.h>
#endif  // _WIN32

#include "ime/common.h"
#include "ime/dict.h"
#include "ime/decoder.h"


int main(int argc, char **argv)
{
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_WTEXT);
    _setmode(_fileno(stdout), _O_WTEXT);
    _setmode(_fileno(stderr), _O_WTEXT);
#else   // _WIN32
    std::setlocale(LC_ALL, "");
#endif  // _WIN32

    std::string dict_file = argv[1];
    std::string model_file = argv[2];

    ime::Dictionary dict(dict_file, 20);
    ime::Decoder decoder(dict);
    decoder.load(model_file);

    while (!std::wcin.eof())
    {
        ime::CodeString code;
        std::wcin >> code;

        std::vector<ime::String> texts;
        std::vector<double> probs;
        if (decoder.predict(code, 10, texts, probs))
        {
            assert(texts.size() == probs.size());

            for (size_t i = 0; i < texts.size(); ++i)
            {
                std::wcout << i + 1 << ": " << texts[i] << ' ' << probs[i] << std::endl;
            }
        }
    }

    return 0;
}
