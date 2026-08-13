// Reports whether a GPU device can be reached, in a process that loads only
// the GPU runtime. rmagine creates its CUDA context while librmagine-cuda is
// being loaded, so on a machine without a device the kernel test aborts before
// main() and cannot report anything itself.
//
// Exit code 0 means at least one device, 77 means none, which is the code
// ctest treats as a skip.

#include <rmcl_ros/util/cuda_to_hip.h>

#include <cstdio>

int main()
{
  int n_devices = 0;
  if(cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices < 1)
  {
    std::printf("no GPU device available\n");
    return 77;
  }
  return 0;
}
