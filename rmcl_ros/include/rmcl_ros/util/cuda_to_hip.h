/*
 * cuda_to_hip.h
 *
 * Compatibility shim that lets the CUDA parts of rmcl_ros build with the
 * ROCm/HIP toolchain (USE_HIP) while leaving the NVIDIA path unchanged.
 *
 * Under HIP it includes the HIP runtime and the hipRAND device header and maps
 * the handful of cuda and curand symbols rmcl_ros uses onto their hip
 * spellings. The sources keep their CUDA spelling and are marked LANGUAGE HIP
 * in CMake. Outside HIP it includes the real CUDA headers, so a CUDA build sees
 * exactly what it saw before.
 *
 * The header is host-includable: hiprand_kernel.h compiles with a plain C++
 * compiler, which is what GladiatorResamplerGPU.cpp relies on.
 */
#ifndef RMCL_ROS_UTIL_CUDA_TO_HIP_H
#define RMCL_ROS_UTIL_CUDA_TO_HIP_H

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// rocrand_mtgp32.h calls printf from a host function without including
// <cstdio> itself, so include it first to keep this header usable whatever the
// surrounding include order is.
#include <cstdio>
#include <hiprand/hiprand_kernel.h>

#define curandState     hiprandState
#define curand_init     hiprand_init
#define curand          hiprand
#define curand_normal   hiprand_normal

#else // CUDA

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#endif // USE_HIP

#endif // RMCL_ROS_UTIL_CUDA_TO_HIP_H
