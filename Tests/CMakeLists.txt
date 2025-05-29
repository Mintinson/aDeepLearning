# Add test executables
add_executable(utils_test "utils_test/test_utils.cpp")
# Link against the metaml target directly (no need for find_package)
target_link_libraries(utils_test PRIVATE metann)

# Add test to CTest
# add_test(NAME utils_test COMMAND utils_test)

# Optional: Group tests into a folder in IDEs like Visual Studio
# set_target_properties(utils_test PROPERTIES FOLDER "Tests")

# If you need any test-specific compile definitions or options
# target_compile_definitions(utils_test PRIVATE 
#    TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
#)
### DATA type tests
add_executable(array_test "datatype_test/test_arr.cpp")
target_link_libraries(array_test PRIVATE metann)

add_executable(matrix_test "datatype_test/test_matrix.cpp")
target_link_libraries(matrix_test PRIVATE metann)

add_executable(dup_batch_test "datatype_test/test_batch_duplicate.cpp")
target_link_libraries(dup_batch_test PRIVATE metann)

add_executable(other_data_test "datatype_test/test_others.cpp")
target_link_libraries(other_data_test PRIVATE metann)

### Unary Operator tests
add_executable(abs_test "unary_operators_test/test_abs.cpp")
target_link_libraries(abs_test PRIVATE metann)

add_executable(transpose_test "unary_operators_test/test_transpose.cpp")
target_link_libraries(transpose_test PRIVATE metann)

add_executable(tanh_test "unary_operators_test/test_tanh.cpp")
target_link_libraries(tanh_test PRIVATE metann)

add_executable(sign_test "unary_operators_test/test_sign.cpp")
target_link_libraries(sign_test PRIVATE metann)

add_executable(softmax_test "unary_operators_test/test_softmax.cpp")
target_link_libraries(softmax_test PRIVATE metann)

add_executable(sigmoid_test "unary_operators_test/test_sigmoid.cpp")
target_link_libraries(sigmoid_test PRIVATE metann)

add_executable(collapse_test "unary_operators_test/test_collapse.cpp")
target_link_libraries(collapse_test PRIVATE metann)

### Binary Operator tests
add_executable(dot_test "binary_operators_test/test_dot.cpp")
target_link_libraries(dot_test PRIVATE metann)

add_executable(add_test "binary_operators_test/test_add.cpp")
target_link_libraries(add_test PRIVATE metann)

add_executable(divide_test "binary_operators_test/test_divide.cpp")
target_link_libraries(divide_test PRIVATE metann)

add_executable(mul_test "binary_operators_test/test_mul.cpp")
target_link_libraries(mul_test PRIVATE metann)

add_executable(sigmoid_derivative "binary_operators_test/test_sigmoid_derivative.cpp")
target_link_libraries(sigmoid_derivative PRIVATE metann)

add_executable(softmax_derivative "binary_operators_test/test_softmax_derivative.cpp")
target_link_libraries(softmax_derivative PRIVATE metann)

add_executable(subtration_test "binary_operators_test/test_subtraction.cpp")
target_link_libraries(subtration_test PRIVATE metann)

add_executable(tanh_derivative_test "binary_operators_test/test_tanh_derivative.cpp")
target_link_libraries(tanh_derivative_test PRIVATE metann)

## Elementart Layer
add_executable(activate_layer_test "elementary_layer/test_activate.cpp")
target_link_libraries(activate_layer_test PRIVATE metann)

add_executable(weight_layer_test "elementary_layer/test_weight_layer.cpp")
target_link_libraries(weight_layer_test PRIVATE metann)

add_executable(filler_test "elementary_layer/test_filler.cpp")
target_link_libraries(filler_test PRIVATE metann)

add_executable(operator_layer_test "elementary_layer/test_oper_layer.cpp")
target_link_libraries(operator_layer_test PRIVATE metann)

add_executable(bias_layer_test "elementary_layer/test_bias.cpp")
target_link_libraries(bias_layer_test PRIVATE metann)