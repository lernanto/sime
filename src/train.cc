/**
 *
 */

#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <locale>
#include <codecvt>
#include <chrono>

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else   // _WIN32
#include <sys/io.h>
#endif  // _WIN32
#include <omp.h>

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

    if (argc < 5)
    {
        ERROR << "usage: "
            << argv[0]
            << " DICT_FILE TRAIN_FILE EVAL_FILE MODEL_FILE [EPOCHS] [BATCH_SIZE] [BEAM_SIZE] [LEARNING_RATE] [THREADS]"
            << std::endl;
        return -1;
    }

    std::string dict_file = argv[1];
    std::string train_file = argv[2];
    std::string eval_file = argv[3];
    std::string model_file = argv[4];

    size_t epochs = 0;
    size_t batch_size = 0;
    size_t beam_size = 0;
    double lr = 0.0;
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
        ss >> beam_size;
    }
    else
    {
        beam_size = 20;
    }
    if (argc >= 9)
    {
        std::stringstream ss(argv[8]);
        ss >> lr;
    }
    else
    {
        lr = 0.01;
    }
    if (argc >= 10)
    {
        std::stringstream ss(argv[9]);
        ss >> threads;
#ifndef _OPENMP
        WARN << "not compiled with OpenMP. thread number has no effect" << std::endl;
#endif  // _OPENMP
    }
    else
    {
        threads = std::min(static_cast<int>(batch_size), 10);
    }

    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    INFO << "train dictionary file = " << conv.from_bytes(dict_file)
        << ", train file = " << conv.from_bytes(train_file)
        << ", evaluation file = " << conv.from_bytes(eval_file)
        << ", epochs = " << epochs
        << ", batch size = " << batch_size
        << ", beam size = " << beam_size
        << ", learning rate = " << lr
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

    ime::Decoder decoder(dict, beam_size, lr);

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
