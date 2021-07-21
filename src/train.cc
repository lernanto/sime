/**
 *
 */

#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <chrono>

#include "omp.h"

#include "ime/common.h"
#include "ime/dict.h"
#include "ime/decoder.h"


int main(int argc, char **argv)
{
    if (argc < 5)
    {
        ERROR << "usage: "
            << argv[0]
            << " DICT_FILE TRAIN_FILE EVAL_FILE [EPOCHS] [BATCH_SIZE] [THREADS]"
            << std::endl;
        return -1;
    }

    std::string dict_file = argv[1];
    std::string train_file = argv[2];
    std::string eval_file = argv[3];
    std::string model_file = argv[4];

    size_t epochs = 0;
    size_t batch_size = 0;
    int threads = 0;
    if (argc >= 6)
    {
        std::stringstream ss(argv[5]);
        ss >> epochs;
    }
    else
    {
        epochs = 2;
    }
    if (argc >= 7)
    {
        std::stringstream ss(argv[6]);
        ss >> batch_size;
    }
    else
    {
        batch_size = 100;
    }
    if (argc >= 8)
    {
        std::stringstream ss(argv[7]);
        ss >> threads;
#ifndef _OPENMP
        WARN << "not compiled with OpenMP. thread number has no effect" << std::endl;
#endif  // _OPENMP
    }
    else
    {
        threads = std::min(static_cast<int>(batch_size), 10);
    }

    INFO << "train dictionary file = " << dict_file
        << ", train file = " << train_file
        << ", evaluation file = " << eval_file
        << ", epochs = " << epochs
        << ", batch size = " << batch_size
        << ", threads = " << threads << std::endl;

#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif  // _OPENMP

    auto start = std::chrono::high_resolution_clock::now();
    ime::Dictionary dict(dict_file, 20);
    auto stop = std::chrono::high_resolution_clock::now();
    INFO << "load dictionary "
        << std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count()
        << "s" << std::endl;

    ime::Decoder decoder(dict);

    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        ime::Metrics metrics;

        start = stop;
        decoder.train(train_file, metrics, batch_size);
        stop = std::chrono::high_resolution_clock::now();

        INFO << "epoch " << epoch + 1 << " train "
            << std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count()
            << "s " << metrics << std::endl;

        metrics.clear();
        start = stop;
        decoder.evaluate(eval_file, metrics, batch_size);
        stop = std::chrono::high_resolution_clock::now();

        INFO << "evaluate "
            << std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count()
            << "s " << metrics << std::endl;
    }

    start = stop;
    decoder.save(model_file);
    stop = std::chrono::high_resolution_clock::now();
    INFO << "save model "
        << std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count()
        << "s" << std::endl;

    return 0;
}
