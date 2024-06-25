#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <ctime>

#include "logger.hpp"

static Logger  embedded = Logger(false, Logger::Level::Message);
static Logger* defaultLog = &embedded;

void SetDefaultLogger(Logger* log) {
    defaultLog = log;
}
Logger* GetDefaultLogger() {
    return defaultLog;
}


Logger::Logger(bool log, Level level, const char* logPath)
      : logPath(logPath)
{
    verbosity = level;
    SetLogging(log);
}

Logger::~Logger() {
    SetLogging(false);
}

Logger::Level& Logger::Verbosity() {
    return verbosity;
}

bool Logger::IsLogging() {
    return log;
}

void Logger::SetLogging(bool log) {
    this->log = log;

    if (!log && logFile != NULL) {
        // Write log end time to file
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];

        time (&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(buffer, sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);

        fprintf(logFile, "*** Log end: %s ***\n", buffer);

        fclose(logFile);
    } else if (log) {
        // logPath can't change, so any file already open is the right one
        if (logFile == NULL) {
            if (logPath != NULL) {
                logFile = fopen(logPath, "a");
            }

            if (logFile == NULL) {
                Warning("Logger::EnableLogging", "Failed to open log file!");
                this->log = false;
            } else {
                // Write log start time to file
                time_t rawtime;
                struct tm * timeinfo;
                char buffer[80];

                time (&rawtime);
                timeinfo = localtime(&rawtime);
                strftime(buffer, sizeof(buffer), "%H:%M:%S", timeinfo);

                fprintf(logFile, "** Log start: %s **\n", buffer);

                fflush(logFile);
            }
        }
    }
}

void Logger::Print(Level type, const char* sender, const char* format, va_list args) {
    static const char* messages[] = { errorMess, warnMess, infoMess, messMess, dbugMess };
    static const int   colors[]   = {         1,        3,        2,        6,        7 };

    if (type <= verbosity) {
        va_list logArgs;
        va_copy(logArgs, args);
        va_list lastArgs;
        va_copy(lastArgs, args);

        printf("\033[1;3%dm[%s] %s: \033[0m", colors[(int)type], sender, messages[(int)type]);
        vprintf(format, args);
        printf("\n");

        if (log && logFile != NULL) {
            time_t rawtime;
            struct tm * timeinfo;
            char buffer[80];

            time (&rawtime);
            timeinfo = localtime(&rawtime);
            strftime(buffer, sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);

            fprintf(logFile, "[%s] ", buffer);

            fprintf(logFile, "[%s] %s: ", sender, messages[(int)type]);
            vfprintf(logFile, format, logArgs);
            fprintf(logFile, "\r\n");

            fflush(logFile);
        }

        char message[128];
        vsnprintf(message, 128, format, lastArgs);
        message[127] = '\0';

        if (messageLog.size() > 512)
            messageLog.erase(messageLog.begin());

        messageLog.push_back(std::string(message));

        va_end(logArgs);
        va_end(lastArgs);
    }
}

void Logger::Print(Level type, const char* sender, const char* format, ...) {
    va_list args;
    va_start(args, format);

    Print(type, sender, format, args);

    va_end(args);
}

void Logger::Error(const char* sender, const char* format, ...) {
    va_list args;
    va_start(args, format);

    Print(Level::Error, sender, format, args);

    va_end(args);
}
void Logger::Warning(const char* sender, const char* format, ...) {
    va_list args;
    va_start(args, format);

    Print(Level::Warning, sender, format, args);

    va_end(args);
}
void Logger::Info(const char* sender, const char* format, ...) {
    va_list args;
    va_start(args, format);

    Print(Level::Info, sender, format, args);

    va_end(args);
}
void Logger::Message(const char* sender, const char* format, ...) {
    va_list args;
    va_start(args, format);

    Print(Level::Message, sender, format, args);

    va_end(args);
}
void Logger::Debug(const char* sender, const char* format, ...) {
    va_list args;
    va_start(args, format);

    Print(Level::Debug, sender, format, args);

    va_end(args);
}

const std::vector<std::string>* Logger::GetMessages() {
    return &messageLog;
}