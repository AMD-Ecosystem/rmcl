
# GPU KERNELS
add_executable(rmcl_ros_tests_gpu_kernels rmcl_gpu_kernels.cpp)

target_link_libraries(rmcl_ros_tests_gpu_kernels
    rmcl_ros_cuda
)

add_test(NAME rmcl_gpu_kernels COMMAND rmcl_ros_tests_gpu_kernels)
