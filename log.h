/**
 * 简单的日志系统.
 */

#ifndef _LOG_H_
#define _LOG_H_

#include <iostream>


#define LOG_VERBOSE 0
#define LOG_DEBUG   1
#define LOG_INFO    2
#define LOG_WARN    3
#define LOG_ERROR   4

#ifndef LOG_LEVEL
#ifndef NDEBUG
#define LOG_LEVEL   LOG_DEBUG
#else
#define LOG_LEVEL   LOG_INFO
#endif  // NDEBUG
#endif  // LOG_LEVEL

#define LOG(level)  if ((level) >= LOG_LEVEL) std::cerr
#define VERBOSE LOG(LOG_VERBOSE) << "[V] " << __FILE__ << ':' << __FUNCTION__ << ':' << __LINE__ << ": "
#define DEBUG   LOG(LOG_DEBUG) << "[D] " << __FILE__ << ':' << __FUNCTION__ << ':' << __LINE__ << ": "
#define INFO    LOG(LOG_INFO) << "[I] "
#define WARN    LOG(LOG_WARN) << "[W] "
#define ERROR   LOG(LOG_ERROR) << "[E] "

#endif  // _LOG_H_
