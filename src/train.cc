/**
 *
 */

#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <chrono>

#include "ime/dict.h"
#include "ime/decoder.h"


int main(int argc, char **argv)
{
    if (argc < 5)
    {
        ERROR << "usage: " << argv[0] << " DICT_FILE, TRAIN_FILE, EVAL_FILE" << std::endl;
        return -1;
    }

    std::string dict_file = argv[1];
    std::string train_file = argv[2];
    std::string eval_file = argv[3];
    std::string model_file = argv[4];

    auto start = std::chrono::high_resolution_clock::now();
    ime::Dictionary dict(dict_file);
    auto stop = std::chrono::high_resolution_clock::now();
    INFO << "load dictionary "
        << std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count()
        << "s" << std::endl;

    ime::Decoder decoder(dict);

    size_t epochs = 2;
    size_t batch_size = 100;
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        std::map<std::string, double> metrics;

        start = stop;
        decoder.train(train_file, metrics, batch_size);
        stop = std::chrono::high_resolution_clock::now();

        INFO << "epoch " << epoch + 1 << " train "
            << std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count()
            << "s" << std::endl;

        for (auto & i : metrics)
        {
            INFO << i.first << " = " << i.second << std::endl;
        }

        metrics.clear();
        start = stop;
        decoder.evaluate(eval_file, metrics);
        stop = std::chrono::high_resolution_clock::now();

        INFO << "evaluate "
            << std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count()
            << "s" << std::endl;

        for (auto & i : metrics)
        {
            INFO << i.first << " = " << i.second << std::endl;
        }
    }

    start = stop;
    decoder.save(model_file);
    stop = std::chrono::high_resolution_clock::now();
    INFO << "save model "
        << std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count()
        << "s" << std::endl;

    return 0;
}
