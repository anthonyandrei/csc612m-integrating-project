#include "simd_support.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

int main(void) {
    SIMDCapabilities caps;
    memset(&caps, 0, sizeof(caps));
    detect_simd_capabilities(&caps);
    print_simd_capabilities(&caps);

    /* Verify basic sanity */
    if (caps.has_cpuid) {
        assert(strlen(caps.vendor) > 0);
        printf("[PASS] CPU vendor identified: %s\n", caps.vendor);
    } else {
        printf("[INFO] Running on non-x86_64 host or emulation layer: %s\n", caps.vendor);
    }
    printf("[PASS] test_simd_support completed successfully.\n");
    return 0;
}
