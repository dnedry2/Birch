/*******************************************************************************
* Copyright 2020-2021 Levi Miller
* 
* Class for display / logging of errors and other messages
*
* Author: Levi Miller
* Date created: 2/6/2021
*******************************************************************************/

#ifndef _BERROR_
#define _BERROR_

#include <stdio.h>
#include <string>
#include <vector>

/*
* Message logging class
* 
* Writes to terminal and optionally a file
*/
class Logger {
public:
    enum class Level {
        Error,   // Critical error
        Warning, // Recoverable error
        Info,    // General information
        Message, // Other
        Debug    // Debug message
    };

    const std::vector<std::string>* GetMessages();

    /*
    * Prints / logs an error
    *
    * Args:
    * sender - function sending message
    * format - output format (like printf)
    * ...    - output
    */
    void Error(const char* sdr, const char* format, ...);
    /*
    * Prints / logs a warning
    *
    * Args:
    * sender - function sending message
    * format - output format (like printf)
    * ...    - output
    */
    void Warning(const char* sdr, const char* format, ...);
    /*
    * Prints / logs info
    *
    * Args:
    * sender - function sending message
    * format - output format (like printf)
    * ...    - output
    */
    void Info(const char* sdr, const char* format, ...);
    /*
    * Prints / logs a message
    *
    * Args:
    * sender - function sending message
    * format - output format (like printf)
    * ...    - output
    */
    void Message(const char* sdr, const char* format, ...);

    /*
    * Prints / logs a debug message
    *
    * Args:
    * sender - function sending message
    * format - output format (like printf)
    * ...    - output
    */
    void Debug(const char* sdr, const char* format, ...);
    /*
    * Generic print / log function
    *
    * Args:
    * type   - message type
    * sender - function sending message
    * format - output format (like printf)
    * ...    - output
    */
    void Print(Level type, const char* sdr, const char* format, ...);
    /*
    * Generic print / log function
    *
    * Args:
    * type   - message type
    * sender - function sending message
    * format - output format (like printf)
    * args   - output args as va_list
    */
    void Print(Level type, const char* sdr, const char* format, va_list args);

    // Current verbosity level
    Level& Verbosity();
    // Logging to file?
    bool IsLogging();
    // Enable / Disable logging
    void SetLogging(bool log);

    /*
    * Logger constructor
    *
    * @param log       - log messages to file
    * @param verbosity - lowest level message to log
    * @param logPath   - path to log file
    */
    Logger(bool log, Logger::Level verbosity, const char* logPath = NULL);
    ~Logger();
private:
    // Location to write log
    const char* logPath;
    // Logfile pointer
    FILE* logFile = NULL;
    // Logging enabled?
    bool log;
    // Verbosity level
    Logger::Level verbosity;
    std::vector<std::string> messageLog;

    const char* errorMess = "Fatal";
    const char* warnMess  = "Uh oh";
    const char* infoMess  = "Info";
    const char* messMess  = "Message";
    const char* dbugMess  = "Debug";
};

void SetDefaultLogger(Logger* log);
Logger* GetDefaultLogger();

#endif

#define DispError_(msg, ...) GetDefaultLogger()->Error(__func__, msg, ##__VA_ARGS__)
#define DispWarning_(msg, ...) GetDefaultLogger()->Warning(__func__, msg, ##__VA_ARGS__)
#define DispInfo_(msg, ...) GetDefaultLogger()->Info(__func__, msg, ##__VA_ARGS__)
#define DispMessage_(msg, ...) GetDefaultLogger()->Message(__func__, msg, ##__VA_ARGS__)

#define DispError(sdr, msg, ...) GetDefaultLogger()->Error(sdr, msg, ##__VA_ARGS__)
#define DispWarning(sdr, msg, ...) GetDefaultLogger()->Warning(sdr, msg, ##__VA_ARGS__)
#define DispInfo(sdr, msg, ...) GetDefaultLogger()->Info(sdr, msg, ##__VA_ARGS__)
#define DispMessage(sdr, msg, ...) GetDefaultLogger()->Message(sdr, msg, ##__VA_ARGS__)

#ifdef DEBUG
    #define DispDebug(msg, ...) GetDefaultLogger()->Debug(__func__, msg, ##__VA_ARGS__)
    #define DebugLine() GetDefaultLogger()->Debug(__func__, "%d : %s", __LINE__, __FILE__)
#else
    #define DispDebug(msg, ...)
    #define DebugLine()
#endif