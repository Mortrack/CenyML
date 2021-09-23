#include "mTimeTer.h"
#include <windows.h>

/**
* The "seconds()" function is used to know the current time
* in seconds with respect to a default-reference time saved
* within the Operative System. Now, what makes this code
* somewhat special is that it detects the current Operative
* System and through that use the code that will work
* accordingly, to make the described function.
* NOTE: Use "mTimeTer.c" works only if you are using Cygwin64
*		terminal or any other of its kind to use ubuntu
*		terminal in windows OS. If you are not using this
*		type of terminal, use "mTime.c" instead. 
*
* @return NULL
*
* @author Unkown
* CREATION DATE: SEPTEMBER 22, 2021
* LAST UPDATE: N/A
*/
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
