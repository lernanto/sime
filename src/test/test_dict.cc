/**
 * @author 黄艺华 (lernanto@foxmail.com)
 * @brief 测试词典
 */

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else   // _WIN32
#include <sys/io.h>
#endif  // _WIN32

#include "ime/dict.h"


bool test_dict(const std::string &fname)
{
    ime::Dictionary dict(fname);
    return true;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_WTEXT);
    _setmode(_fileno(stdout), _O_WTEXT);
    _setmode(_fileno(stderr), _O_WTEXT);
#else   // _WIN32
    std::setlocale(LC_ALL, "");
#endif  // _WIN32

    return test_dict(argv[1]) ? 0 : -1;
}
