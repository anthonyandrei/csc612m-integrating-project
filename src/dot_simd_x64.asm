; =============================================================================
; Single-Thread 256-Bit AVX2 SIMD Dot-Product Kernel (Issue #5)
; CSC612M: Advanced Computer Architecture, De La Salle University
;
; Computes the dot product of two single-precision float arrays using AVX2.
; Unrolls 8 floats (256 bits) per iteration, performs horizontal reduction,
; and handles any arbitrary leftover components with a scalar remainder loop.
;
; C prototype:
;   float dot_product_simd_ymm(const float* a, const float* b, uint32_t dim);
;
; ABI Parameter Mapping:
;   Windows x64: RCX = a, RDX = b, R8D = dim -> Return: XMM0
;   System V:    RDI = a, RSI = b, EDX = dim -> Return: XMM0
; =============================================================================

default rel

section .text
global dot_product_simd_ymm
global _dot_product_simd_ymm

%ifidn __OUTPUT_FORMAT__, win64
    %define PTR_A   rcx
    %define PTR_B   rdx
    %define DIM_REG r8d
%else
    %define PTR_A   rdi
    %define PTR_B   rsi
    %define DIM_REG edx
%endif

dot_product_simd_ymm:
_dot_product_simd_ymm:
    ; 1. Null pointer and boundary verification
    test DIM_REG, DIM_REG
    jle .zero_dim
    test PTR_A, PTR_A
    jz .zero_dim
    test PTR_B, PTR_B
    jz .zero_dim

    ; 2. Compute 8-float vector batch limit: r10d = (dim / 8) * 8
    mov r10d, DIM_REG
    shr r10d, 3
    shl r10d, 3             ; r10d = count of floats in full 256-bit batches

    ; 3. Initialize 256-bit vector accumulator
    vxorps ymm4, ymm4, ymm4
    xor eax, eax            ; eax = float index counter

    ; 4. 256-bit AVX2 loop (8 single-precision floats / 32 bytes per iteration)
.batch_loop:
    cmp eax, r10d
    jge .reduce_vector

    vmovups ymm0, [PTR_A + rax * 4]
    vmovups ymm1, [PTR_B + rax * 4]
    vmulps ymm2, ymm0, ymm1
    vaddps ymm4, ymm4, ymm2

    add eax, 8
    jmp .batch_loop

.reduce_vector:
    ; 5. Horizontal reduction: reduce 8-lane YMM accumulator into 32-bit scalar
    ; Extract upper 128-bit lane from ymm4 into xmm3
    vextractf128 xmm3, ymm4, 1
    vaddps xmm4, xmm4, xmm3

    ; Pairwise horizontal add within 128-bit half (xmm4)
    vhaddps xmm4, xmm4, xmm4
    vhaddps xmm4, xmm4, xmm4
    ; Lowest 32-bit lane of xmm4 now holds the vector batches sum

.remainder_loop:
    ; 6. Remainder / tail loop for leftover elements: 1 float per iteration
    cmp eax, DIM_REG
    jge .done

    vmovss xmm0, [PTR_A + rax * 4]
    vmovss xmm1, [PTR_B + rax * 4]
    vmulss xmm2, xmm0, xmm1
    vaddss xmm4, xmm4, xmm2

    inc eax
    jmp .remainder_loop

.done:
    vmovaps xmm0, xmm4
    ret

.zero_dim:
    vxorps xmm0, xmm0, xmm0
    ret
