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
    if (argc < 5)
    {
        ERROR << "usage: " << argv[0] << " DICT_FILE, TRAIN_FILE, EVAL_FILE" << std::endl;
        return -1;
    }

    std::string dict_file = argv[1];
    std::string train_file = argv[2];
    std::string eval_file = argv[3];
    std::string model_file = argv[4];

    ime::Dictionary dict(dict_file);
    ime::Decoder decoder(dict);

    size_t epochs = 2;
    size_t batch_size = 100;
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        std::map<std::string, double> metrics;
        decoder.train(train_file, metrics, batch_size);
        INFO << "epoch " << epoch + 1 << " train" << std::endl;
        for (auto & i : metrics)
        {
            INFO << i.first << " = " << i.second << std::endl;
        }

        metrics.clear();
        decoder.evaluate(eval_file, metrics);
        INFO << "evaluate" << std::endl;
        for (auto & i : metrics)
        {
            INFO << i.first << " = " << i.second << std::endl;
        }
    }

    decoder.save(model_file);
    return 0;
}
