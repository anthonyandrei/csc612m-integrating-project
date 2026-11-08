#include "simd_support.h"
#include <stdio.h>
#include <string.h>

#if defined(_MSC_VER)
    #include <intrin.h>
    #include <windows.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <cpuid.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
static void get_cpuid(int leaf, int subleaf, int info[4]) {
#if defined(_MSC_VER)
    __cpuidex(info, leaf, subleaf);
#else
    __cpuid_count(leaf, subleaf, info[0], info[1], info[2], info[3]);
#endif
}

static uint64_t get_xcr0(void) {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif !defined(__apple_build_version__)
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#else
    return 0;
#endif
}
#endif

void detect_simd_capabilities(SIMDCapabilities* caps) {
    memset(caps, 0, sizeof(*caps));

#if defined(__x86_64__) || defined(_M_X64)
    caps->has_cpuid = 1;
    int info[4];

    /* Leaf 0: Vendor String & Max Leaf */
    get_cpuid(0, 0, info);
    int max_leaf = info[0];
    memcpy(caps->vendor + 0, &info[1], 4);
    memcpy(caps->vendor + 4, &info[3], 4);
    memcpy(caps->vendor + 8, &info[2], 4);
    caps->vendor[12] = '\0';

    /* Leaf 1: Basic Features (OSXSAVE, AVX) */
    if (max_leaf >= 1) {
        get_cpuid(1, 0, info);
        caps->has_os_xsave = (info[2] & (1 << 27)) != 0;
        caps->has_avx = (info[2] & (1 << 28)) != 0;
    }

    /* Leaf 7: Extended Features (AVX2) */
    if (max_leaf >= 7) {
        get_cpuid(7, 0, info);
        caps->has_avx2 = (info[1] & (1 << 5)) != 0;
    }

    /* Check OS YMM state enablement via XGETBV or Win32 API */
    if (caps->has_os_xsave) {
#if defined(_WIN32)
        if (IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE)) {
            caps->has_os_ymm_support = 1;
        } else {
            uint64_t xcr0 = get_xcr0();
            caps->has_os_ymm_support = ((xcr0 & 0x06) == 0x06);
        }
#else
        uint64_t xcr0 = get_xcr0();
        caps->has_os_ymm_support = ((xcr0 & 0x06) == 0x06);
#endif
    }

    /* Brand string from 0x80000002 - 0x80000004 */
    get_cpuid(0x80000000, 0, info);
    unsigned int max_ext = (unsigned int)info[0];
    if (max_ext >= 0x80000004) {
        int brand[12];
        get_cpuid(0x80000002, 0, &brand[0]);
        get_cpuid(0x80000003, 0, &brand[4]);
        get_cpuid(0x80000004, 0, &brand[8]);
        memcpy(caps->brand, brand, 48);
        caps->brand[48] = '\0';
    }

    caps->is_simd_supported = caps->has_avx2 && caps->has_os_ymm_support;
#else
    /* Non-x86_64 host (e.g. ARM64 development fallback) */
    caps->has_cpuid = 0;
    caps->has_avx = 0;
    caps->has_avx2 = 0;
    caps->has_os_ymm_support = 0;
    caps->is_simd_supported = 0;
    snprintf(caps->vendor, sizeof(caps->vendor), "Non-x86_64");
    snprintf(caps->brand, sizeof(caps->brand), "Architecture without native AVX2");
#endif
}

void print_simd_capabilities(const SIMDCapabilities* caps) {
    printf("--- Target CPU & Vector Diagnostics (AC 1) ---\n");
    printf("  CPU Vendor:            %s\n", caps->vendor);
    if (caps->brand[0])
        printf("  CPU Brand:             %s\n", caps->brand);
    printf("  CPUID Supported:       %s\n", caps->has_cpuid ? "YES" : "NO");
    printf("  AVX Supported (CPU):   %s\n", caps->has_avx ? "YES" : "NO");
    printf("  AVX2 Supported (CPU):  %s\n", caps->has_avx2 ? "YES" : "NO");
    printf("  OSXSAVE Active (CPU):  %s\n", caps->has_os_xsave ? "YES" : "NO");
    printf("  OS YMM State Support:  %s\n", caps->has_os_ymm_support ? "YES (XCR0[2:1]=11b)" : "NO");
    printf("  Single-Thread SIMD:    %s\n", caps->is_simd_supported ? "READY (AVX2 enabled)" : "NOT AVAILABLE");
    printf("-----------------------------------------------\n");
}
