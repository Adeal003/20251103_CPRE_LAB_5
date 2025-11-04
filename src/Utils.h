#pragma once

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "Config.h"

// Check for string formatting support
// #ifdef __has_include
// #  if __has_include(<format>)
// #    include <format>
// #    include <string_view>
// using std::format;
// using std::make_format_args;
// #  elif __has_include(<fmt/core.h>)
// //#define FMT_HEADER_ONLY
// #    include <fmt/core.h>
// using fmt::make_format_args;
// using fmt::vformat;
// #  else
// #    define STR_FORMAT_NONE
// #  endif
// #else
// #endif
// Don't need string format
#define STR_FORMAT_NONE


#ifdef ZEDBOARD
#   include <ff.h>
#   include <xtime_l.h>
#endif

namespace ML {

// --- Color Logging Codes ---
enum CCode {
    RESET = 0,
    BRIGHT = 1,
    UNDERLINE = 4,
    INVERSE = 7,
    BRIGHT_OFF = 21,
    UNDERLINE_OFF = 24,
    INVERSE_OFF = 27,

    FG_BLACK = 30,
    FG_RED = 31,
    FG_GREEN = 32,
    FG_YELLOW = 33,
    FG_BLUE = 34,
    FG_MAGENTA = 35,
    FG_CYAN = 36,
    FG_WHITE = 37,
    FG_DEFAULT = 39,

    BG_BLACK = 40,
    BG_RED = 41,
    BG_GREEN = 42,
    BG_YELLOW = 43,
    BG_BLUE = 44,
    BG_MAGENTA = 45,
    BG_CYAN = 46,
    BG_WHITE = 47,
    BG_DEFAULT = 49
};

// --- Logging helper ---
class LogMod {
   private:
    CCode code;

   public:
    LogMod(CCode pCode) : code(pCode) {}
    friend std::ostream& operator<<(std::ostream& os, const LogMod& mod) { return os << "\033[" << mod.code << "m"; }
};

// TODO: Use fmt lib formatting for colors and other styles instead if available
// Log a non-decorated message
#ifndef STR_FORMAT_NONE
template <typename... Args> inline void log(const std::string_view& fmt_str, Args&&... args) {
    std::string msg = vformat(fmt_str, make_format_args(args...));
#else
template <typename... Args> inline void log(const std::string& msg, Args&&... args) {
#endif
    if (Config::FANCY_LOGGING)
        std::cout << msg << std::endl;
    else
        std::cout << msg << std::endl;
}

// Log a info Message
#ifndef STR_FORMAT_NONE
template <typename... Args> inline void logInfo(const std::string_view& fmt_str, Args&&... args) {
    std::string msg = vformat(fmt_str, make_format_args(args...));
#else
template <typename... Args> inline void logInfo(const std::string& msg, Args&&... args) {
#endif
    if (Config::FANCY_LOGGING)
        std::cout << "[" << LogMod(CCode::FG_CYAN) << "Info" << LogMod(CCode::FG_DEFAULT) << "]: " << msg << std::endl;
    else
        std::cout << "[Info]: " << msg << std::endl;
}

// Log a debug Message
#ifndef STR_FORMAT_NONE
template <typename... Args> inline void logDebug(const std::string_view& fmt_str, Args&&... args) {
    std::string msg = vformat(fmt_str, make_format_args(args...));
#else
template <typename... Args> inline void logDebug(const std::string& msg, Args&&... args) {
#endif
    if (Config::FANCY_LOGGING)
        std::cout << "[" << LogMod(CCode::FG_GREEN) << "Debug" << LogMod(CCode::FG_DEFAULT) << "]: " << msg << std::endl;
    else
        std::cout << "[Debug]: " << msg << std::endl;
}

// Log a warning Message
#ifndef STR_FORMAT_NONE
template <typename... Args> inline void logWarn(const std::string_view& fmt_str, Args&&... args) {
    std::string msg = vformat(fmt_str, make_format_args(args...));
#else
template <typename... Args> inline void logWarn(const std::string& msg, Args&&... args) {
#endif
    if (Config::FANCY_LOGGING)
        std::cout << "[" << LogMod(CCode::FG_YELLOW) << "Warning" << LogMod(CCode::FG_DEFAULT) << "]: " << msg << std::endl;
    else
        std::cout << "[Warning]: " << msg << std::endl;
}

// Log an error Message
#ifndef STR_FORMAT_NONE
template <typename... Args> inline void logError(const std::string_view& fmt_str, Args&&... args) {
    std::string msg = vformat(fmt_str, make_format_args(args...));
#else
template <typename... Args> inline void logError(const std::string& msg, Args&&... args) {
#endif
    if (Config::FANCY_LOGGING)
        std::cerr << "[" << LogMod(CCode::FG_RED) << "Error" << LogMod(CCode::FG_DEFAULT) << "]: " << msg << std::endl;
    else
        std::cerr << "[Error]: " << msg << std::endl;
}

// --- Timing Functions ---
class Timer {
   public:
#ifdef ZEDBOARD
    XTime begin, end;
#else
    std::chrono::time_point<std::chrono::steady_clock> begin, end;
#endif
    std::string name;
    float milliseconds;

    Timer(const std::string&& name): name(name) {}
    ~Timer() {}

    // Start the Timer
    void start() {
#ifndef DISABLE_TIMING
#ifdef ZEDBOARD
        XTime_GetTime(&begin);
#else
        begin = std::chrono::steady_clock::now();
#endif
#endif
    }

    // Stop the Timer
    void stop() {
#ifndef DISABLE_TIMING
#ifdef ZEDBOARD
        XTime_GetTime(&end);
        milliseconds = ((end - begin) * 1000) / COUNTS_PER_SECOND;
#else
        end = std::chrono::steady_clock::now();
        milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0f;
#endif
        log("Timer " + name + ": elapsed=" + std::to_string(milliseconds) + "ms");
#endif
    }
};

class Path : public std::string {
    public:
    inline Path(const std::string&& str): std::string(str) {}
    inline Path(const char* str): std::string(str) {}

    inline Path operator/(const std::string&& other) const {
        return Path(*this + "/" + other);
    }

    inline Path operator/(const char* other) const {
        return Path(*this + "/" + other);
    }
};

template<typename T>
T clamp(T value, T min, T max) {
    return value < min ? min : (value > max ? max : value);
}

}