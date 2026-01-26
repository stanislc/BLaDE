/**
 * Standalone BLaDE logging implementation.
 *
 * This file is ONLY compiled for standalone BLaDE builds.
 * CHARMM provides its own implementation via blade_api.
 */

#include <cstdio>

extern "C" void blade_log(const char* message) {
    if (message) {
        printf("%s", message);
        fflush(stdout);
    }
}
