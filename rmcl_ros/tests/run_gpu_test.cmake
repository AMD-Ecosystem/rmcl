# Runs the GPU kernel test where a device is available and reports a skip where
# none is, so that a build host without a GPU does not turn a missing device
# into a test failure. Only probe exit 77 means no device; a probe that fails
# for any other reason is reported as a failure, so that a test which never ran
# is not mistaken for a machine without a GPU. Invoked by ctest, see
# tests/CMakeLists.txt.

execute_process(COMMAND ${PROBE} RESULT_VARIABLE probe_result)

if(probe_result EQUAL 77)
    message(FATAL_ERROR "no GPU device available, skipping")
elseif(NOT probe_result EQUAL 0)
    message(FATAL_ERROR "device probe exited with ${probe_result}")
endif()

execute_process(COMMAND ${TEST} RESULT_VARIABLE test_result)

if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "${TEST} exited with ${test_result}")
endif()
