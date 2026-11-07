#ifndef SIMD_SUPPORT_H
#define SIMD_SUPPORT_H

#include <stdint.h>

/**
 * Diagnostics and capability flags for CPU and operating system vector support (AC 1).
 */
typedef struct {
    int has_cpuid;           /**< CPUID instruction is supported. */
    int has_avx;             /**< CPU supports 256-bit AVX. */
    int has_avx2;            /**< CPU supports AVX2 integer/vector extensions. */
    int has_os_xsave;        /**< CPU supports XSAVE and OS enabled OSXSAVE. */
    int has_os_ymm_support;  /**< Operating system manages and restores 256-bit YMM state. */
    int is_simd_supported;   /**< True if both CPU and OS support 256-bit AVX2. */
    char vendor[13];         /**< CPU vendor string (e.g., GenuineIntel, AuthenticAMD). */
    char brand[49];          /**< CPU brand text (e.g., Intel Core i7 / AMD Ryzen). */
} SIMDCapabilities;

/**
 * Detects CPU and operating system vector capabilities using CPUID and XGETBV.
 * On Windows, also verifies via IsProcessorFeaturePresent.
 *
 * @param caps Pointer to structure receiving the capability flags.
 */
void detect_simd_capabilities(SIMDCapabilities* caps);

/**
 * Prints formatted diagnostics of the CPU and OS vector state to stdout.
 *
 * @param caps Pointer to populated capabilities.
 */
void print_simd_capabilities(const SIMDCapabilities* caps);

#endif /* SIMD_SUPPORT_H */
