#ifndef RMCL_ROS_MCL_RESAMPLING_CUH
#define RMCL_ROS_MCL_RESAMPLING_CUH

/**
 * @brief collection of functions for resampling
 */

#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/types/shared_functions.h>

#include <rmcl_ros/rmcl/ParticleAttributes.hpp>

#include <rmcl_ros/rmcl/GladiatorResamplerConfig.hpp>

#include "Resampler.hpp"

#include <rmcl_ros/util/cuda_to_hip.h>


namespace rmcl
{

void init_curand(rmagine::MemoryView<curandState, rmagine::VRAM_CUDA>& curand_states);

struct SimpleLikelihoodStats
{
  // No default member initializers here: the reduction kernel declares a
  // __shared__ array of this type, and a shared variable cannot be
  // initialized. Both members are written before they are read.
  float sum;
  float max;
};

void compute_stats(
  const rmagine::MemoryView<rmagine::Transform, rmagine::VRAM_CUDA>& particle_poses,
  const rmagine::MemoryView<ParticleAttributes, rmagine::VRAM_CUDA>& particle_attrs,
  rmagine::MemoryView<SimpleLikelihoodStats, rmagine::VRAM_CUDA> stats);

void gladiator_resample(
  const rmagine::MemoryView<rmagine::Transform, rmagine::VRAM_CUDA> particle_poses,
  const rmagine::MemoryView<ParticleAttributes, rmagine::VRAM_CUDA> particle_attrs,
  const rmagine::MemoryView<SimpleLikelihoodStats, rmagine::VRAM_CUDA> stats,
  rmagine::MemoryView<curandState, rmagine::VRAM_CUDA> rstates,
  rmagine::MemoryView<rmagine::Transform, rmagine::VRAM_CUDA> particle_poses_new,
  rmagine::MemoryView<ParticleAttributes, rmagine::VRAM_CUDA> particle_attrs_new,
  const GladiatorResamplerConfig& config);

} // namespace rmcl

#endif // RMCL_ROS_MCL_RESAMPLING_CUH