// Correctness gate for the GPU kernels of rmcl_ros: the particle motion
// update, the likelihood reduction and the gladiator resampler. Every kernel
// result is compared against a CPU reference computed here, so the test says
// something about the GPU and not only about the build.
//
// The random number generator differs between CUDA and ROCm, so the resampler
// is checked through invariants that hold for any stream (structural
// copy-through, no NaN, the distribution of the drawn normal samples and
// run-to-run determinism) instead of against a recorded sequence.

#include <rmcl_ros/rmcl/particle_motion.cuh>
#include <rmcl_ros/rmcl/resampling.cuh>

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rm = rmagine;

namespace
{

// deliberately not a multiple of the block size
constexpr unsigned int N = 4099;

void require(bool cond, const std::string& msg)
{
  if(!cond)
  {
    throw std::runtime_error("FAIL: " + msg);
  }
}

void check_rel(double got, double ref, double eps, const std::string& what)
{
  const double denom = std::max(1.0, std::abs(ref));
  const double rel = std::abs(got - ref) / denom;
  if(!(rel <= eps))
  {
    throw std::runtime_error("FAIL: " + what + " got=" + std::to_string(got)
      + " ref=" + std::to_string(ref) + " rel=" + std::to_string(rel));
  }
}

bool bitwise_equal(const void* a, const void* b, size_t bytes)
{
  return std::memcmp(a, b, bytes) == 0;
}

rm::Transform make_pose(unsigned int i)
{
  const double p = static_cast<double>(i) / static_cast<double>(N);
  rm::Transform T;
  T.t.x = static_cast<float>(std::sin(p * 7.0) * 3.0);
  T.t.y = static_cast<float>(p * 4.0 - 2.0);
  T.t.z = static_cast<float>(std::cos(p * 5.0));
  rm::EulerAngles e;
  e.roll  = static_cast<float>(std::sin(p * 3.0));
  e.pitch = static_cast<float>(std::cos(p * 2.0) * 0.5);
  e.yaw   = static_cast<float>(p * 2.0 - 1.0);
  T.R.set(e);
  return T;
}

void test_particle_move_and_forget()
{
  rm::Memory<rm::Transform, rm::RAM> poses(N);
  rm::Memory<rmcl::ParticleAttributes, rm::RAM> attrs(N);

  for(unsigned int i = 0; i < N; i++)
  {
    poses[i] = make_pose(i);
    attrs[i].likelihood.mean = static_cast<float>(i % 97) / 97.0f;
    attrs[i].likelihood.sigma = 0.5f;
    attrs[i].likelihood.n_meas = 100 + i % 1000;
    attrs[i].state_sigma = rm::Matrix_<float, 6, 1>::Zeros();
  }

  rm::Transform T_bnew_bold;
  T_bnew_bold.t = {0.25f, -0.5f, 0.75f};
  rm::EulerAngles e;
  e.roll = 0.1f; e.pitch = -0.2f; e.yaw = 0.3f;
  T_bnew_bold.R.set(e);

  const double forget_rate = 0.125;

  rm::Memory<rm::Transform, rm::VRAM_CUDA> poses_gpu = poses;
  rm::Memory<rmcl::ParticleAttributes, rm::VRAM_CUDA> attrs_gpu = attrs;

  rmcl::particle_move_and_forget(poses_gpu, attrs_gpu, T_bnew_bold, forget_rate);

  rm::Memory<rm::Transform, rm::RAM> poses_res = poses_gpu;
  rm::Memory<rmcl::ParticleAttributes, rm::RAM> attrs_res = attrs_gpu;

  for(unsigned int i = 0; i < N; i++)
  {
    const rm::Transform ref = poses[i] * T_bnew_bold;
    check_rel(poses_res[i].t.x, ref.t.x, 1e-5, "move t.x");
    check_rel(poses_res[i].t.y, ref.t.y, 1e-5, "move t.y");
    check_rel(poses_res[i].t.z, ref.t.z, 1e-5, "move t.z");
    check_rel(poses_res[i].R.x, ref.R.x, 1e-5, "move R.x");
    check_rel(poses_res[i].R.y, ref.R.y, 1e-5, "move R.y");
    check_rel(poses_res[i].R.z, ref.R.z, 1e-5, "move R.z");
    check_rel(poses_res[i].R.w, ref.R.w, 1e-5, "move R.w");

    const uint32_t n_meas = attrs[i].likelihood.n_meas;
    const uint32_t n_meas_ref =
      static_cast<uint32_t>(n_meas - forget_rate * static_cast<double>(n_meas));
    require(attrs_res[i].likelihood.n_meas == n_meas_ref,
      "forgotten n_meas at " + std::to_string(i));
    require(attrs_res[i].likelihood.mean == attrs[i].likelihood.mean,
      "likelihood mean untouched at " + std::to_string(i));
  }

  std::cout << "particle_move_and_forget: OK (" << N << " particles)" << std::endl;
}

void test_compute_stats()
{
  rm::Memory<rm::Transform, rm::RAM> poses(N);
  rm::Memory<rmcl::ParticleAttributes, rm::RAM> attrs(N);

  double ref_sum = 0.0;
  float ref_max = 0.0f;

  for(unsigned int i = 0; i < N; i++)
  {
    poses[i] = make_pose(i);
    const float L = static_cast<float>(std::sin(static_cast<double>(i) * 0.37) * 0.5 + 0.5);
    attrs[i].likelihood.mean = L;
    attrs[i].likelihood.sigma = 1.0f;
    attrs[i].likelihood.n_meas = 1;
    attrs[i].state_sigma = rm::Matrix_<float, 6, 1>::Zeros();
    ref_sum += L;
    ref_max = std::max(ref_max, L);
  }

  rm::Memory<rm::Transform, rm::VRAM_CUDA> poses_gpu = poses;
  rm::Memory<rmcl::ParticleAttributes, rm::VRAM_CUDA> attrs_gpu = attrs;
  // one output block: the kernel indexes the input by N * blockIdx.x
  rm::Memory<rmcl::SimpleLikelihoodStats, rm::VRAM_CUDA> stats_gpu(1);

  rmcl::compute_stats(poses_gpu, attrs_gpu, stats_gpu);
  rm::Memory<rmcl::SimpleLikelihoodStats, rm::RAM> stats = stats_gpu;

  check_rel(stats[0].sum, ref_sum, 1e-5, "stats sum");
  require(stats[0].max == ref_max, "stats max exact");

  // the reduction must not depend on the order the wavefronts happen to run in
  rmcl::compute_stats(poses_gpu, attrs_gpu, stats_gpu);
  rm::Memory<rmcl::SimpleLikelihoodStats, rm::RAM> stats2 = stats_gpu;
  require(bitwise_equal(&stats[0], &stats2[0], sizeof(rmcl::SimpleLikelihoodStats)),
    "stats reduction is deterministic");

  std::cout << "compute_stats: OK (sum=" << stats[0].sum
            << " max=" << stats[0].max << ")" << std::endl;
}

void test_gladiator_resample()
{
  rm::Memory<rm::Transform, rm::RAM> poses(N);
  rm::Memory<rmcl::ParticleAttributes, rm::RAM> attrs(N);

  for(unsigned int i = 0; i < N; i++)
  {
    poses[i] = make_pose(i);
    // unique likelihoods: the winner of a duel can be identified from the
    // resampled attributes alone, which is what lets the noise be recovered
    attrs[i].likelihood.mean = static_cast<float>(i + 1);
    attrs[i].likelihood.sigma = 1.0f;
    attrs[i].likelihood.n_meas = 1000;
    attrs[i].state_sigma = rm::Matrix_<float, 6, 1>::Zeros();
  }

  rmcl::GladiatorResamplerConfig config;
  config.min_noise_tx = 1.0f;
  config.min_noise_ty = 1.0f;
  config.min_noise_tz = 1.0f;
  config.min_noise_roll = 0.0f;
  config.min_noise_pitch = 0.0f;
  config.min_noise_yaw = 0.0f;
  // no forgetting, so n_meas has to survive the resampling untouched
  config.likelihood_forget_per_meter = 0.0f;
  config.likelihood_forget_per_radian = 0.0f;

  rm::Memory<rm::Transform, rm::VRAM_CUDA> poses_gpu = poses;
  rm::Memory<rmcl::ParticleAttributes, rm::VRAM_CUDA> attrs_gpu = attrs;
  rm::Memory<rm::Transform, rm::VRAM_CUDA> poses_new_gpu(N);
  rm::Memory<rmcl::ParticleAttributes, rm::VRAM_CUDA> attrs_new_gpu(N);
  rm::Memory<rmcl::SimpleLikelihoodStats, rm::VRAM_CUDA> stats_gpu(1);
  rm::Memory<curandState, rm::VRAM_CUDA> rstates(N);

  rmcl::compute_stats(poses_gpu, attrs_gpu, stats_gpu);

  rmcl::init_curand(rstates);
  rmcl::gladiator_resample(poses_gpu, attrs_gpu, stats_gpu, rstates,
    poses_new_gpu, attrs_new_gpu, config);

  rm::Memory<rm::Transform, rm::RAM> poses_new = poses_new_gpu;
  rm::Memory<rmcl::ParticleAttributes, rm::RAM> attrs_new = attrs_new_gpu;

  std::vector<double> samples;
  unsigned int n_replaced = 0;

  for(unsigned int i = 0; i < N; i++)
  {
    const rm::Transform& out = poses_new[i];
    require(std::isfinite(out.t.x) && std::isfinite(out.t.y) && std::isfinite(out.t.z),
      "finite translation at " + std::to_string(i));
    require(std::isfinite(out.R.x) && std::isfinite(out.R.y)
      && std::isfinite(out.R.z) && std::isfinite(out.R.w),
      "finite rotation at " + std::to_string(i));
    require(std::isfinite(attrs_new[i].likelihood.mean), "finite mean at " + std::to_string(i));

    const bool pose_kept = bitwise_equal(&poses_new[i], &poses[i], sizeof(rm::Transform));
    const bool attr_kept = bitwise_equal(&attrs_new[i], &attrs[i], sizeof(rmcl::ParticleAttributes));

    if(pose_kept && attr_kept)
    {
      // duel lost by the challenger: the particle must be copied through
      // untouched, not merely reproduced approximately
      continue;
    }

    require(attrs_new[i].likelihood.mean > attrs[i].likelihood.mean,
      "a replaced particle carries a stronger likelihood at " + std::to_string(i));
    require(attrs_new[i].likelihood.n_meas == 1000,
      "n_meas survives a resampling without forgetting at " + std::to_string(i));

    // the winner is the particle whose likelihood was copied in
    const unsigned int winner = static_cast<unsigned int>(attrs_new[i].likelihood.mean) - 1;
    require(winner < N, "winner index in range at " + std::to_string(i));
    samples.push_back(out.t.x - poses[winner].t.x);
    samples.push_back(out.t.y - poses[winner].t.y);
    samples.push_back(out.t.z - poses[winner].t.z);
    n_replaced++;
  }

  require(n_replaced > N / 10, "the resampler actually replaced particles");

  // with unit noise the recovered offsets are the drawn normal samples
  double mean = 0.0;
  for(double s : samples) { mean += s; }
  mean /= static_cast<double>(samples.size());
  double var = 0.0;
  for(double s : samples) { var += (s - mean) * (s - mean); }
  var /= static_cast<double>(samples.size() - 1);
  const double sd = std::sqrt(var);

  require(std::abs(mean) < 0.1, "normal samples are centred, got " + std::to_string(mean));
  require(std::abs(sd - 1.0) < 0.1, "normal samples have unit spread, got " + std::to_string(sd));

  // same seeds, same result
  rm::Memory<curandState, rm::VRAM_CUDA> rstates2(N);
  rm::Memory<rm::Transform, rm::VRAM_CUDA> poses_new2_gpu(N);
  rm::Memory<rmcl::ParticleAttributes, rm::VRAM_CUDA> attrs_new2_gpu(N);
  rmcl::init_curand(rstates2);
  rmcl::gladiator_resample(poses_gpu, attrs_gpu, stats_gpu, rstates2,
    poses_new2_gpu, attrs_new2_gpu, config);
  rm::Memory<rm::Transform, rm::RAM> poses_new2 = poses_new2_gpu;
  rm::Memory<rmcl::ParticleAttributes, rm::RAM> attrs_new2 = attrs_new2_gpu;

  require(bitwise_equal(poses_new.raw(), poses_new2.raw(), N * sizeof(rm::Transform)),
    "resampled poses are reproducible");
  require(bitwise_equal(attrs_new.raw(), attrs_new2.raw(), N * sizeof(rmcl::ParticleAttributes)),
    "resampled attributes are reproducible");

  std::cout << "gladiator_resample: OK (" << n_replaced << " of " << N
            << " particles replaced, noise mean=" << mean << " sd=" << sd << ")" << std::endl;
}

} // namespace

int main()
{
  try
  {
    test_particle_move_and_forget();
    test_compute_stats();
    test_gladiator_resample();
  }
  catch(const std::exception& ex)
  {
    std::cerr << ex.what() << std::endl;
    return 1;
  }

  std::cout << "all rmcl gpu kernel tests passed" << std::endl;
  return 0;
}
