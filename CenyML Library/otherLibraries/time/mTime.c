#include "mTime.h"

/**
* The "seconds()" function is used to know the current time
* in seconds with respect to a default-reference time saved
* within the Operative System. Now, what makes this code
* somewhat special is that it detects the current Operative
* System and through that use the code that will work
* accordingly, to make the described function.
* NOTE: Use mTimeTer.c instead if you are using Cygwin64
*		terminal or any other of its kind to use ubuntu
*		terminal in windows OS. Otherwise, you will get
*		an error during code compilation. 
*
* @return NULL
*
* @author Unkown
* CREATION DATE: SEPTEMBER 22, 2021
* LAST UPDATE: N/A
*/
#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static double seconds(void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else {
        return (double)GetTickCount() / 1000.0;
    }
}
#elif defined(__linux__) || defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
static double seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#else
#error unsupported platform
#endif

