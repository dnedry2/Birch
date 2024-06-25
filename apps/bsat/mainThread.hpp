#ifndef __B_MAIN_THREAD__
#define __B_MAIN_THREAD__

struct MainThreadTask {
    void (*Task)(void*);
    void* args;

    MainThreadTask(void (*task)(void*), void* args) : Task(task), args(args) { }
};

#endif