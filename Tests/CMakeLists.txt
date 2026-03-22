function(add_metann_test TARGET_NAME SOURCE_FILE)
	add_executable(${TARGET_NAME} ${SOURCE_FILE})
	target_link_libraries(${TARGET_NAME} PRIVATE metann)
	add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endfunction()

### Utils tests
add_metann_test(utils_test "utils_test/test_utils.cpp")

### Data type tests
add_metann_test(array_test "datatype_test/test_arr.cpp")
add_metann_test(matrix_test "datatype_test/test_matrix.cpp")
add_metann_test(dup_batch_test "datatype_test/test_batch_duplicate.cpp")
add_metann_test(other_data_test "datatype_test/test_others.cpp")

### Unary operator tests
add_metann_test(abs_test "unary_operators_test/test_abs.cpp")
add_metann_test(transpose_test "unary_operators_test/test_transpose.cpp")
add_metann_test(tanh_test "unary_operators_test/test_tanh.cpp")
add_metann_test(sign_test "unary_operators_test/test_sign.cpp")
add_metann_test(softmax_test "unary_operators_test/test_softmax.cpp")
add_metann_test(sigmoid_test "unary_operators_test/test_sigmoid.cpp")
add_metann_test(collapse_test "unary_operators_test/test_collapse.cpp")

### Binary operator tests
add_metann_test(dot_test "binary_operators_test/test_dot.cpp")
add_metann_test(add_test "binary_operators_test/test_add.cpp")
add_metann_test(divide_test "binary_operators_test/test_divide.cpp")
add_metann_test(mul_test "binary_operators_test/test_mul.cpp")
add_metann_test(sigmoid_derivative "binary_operators_test/test_sigmoid_derivative.cpp")
add_metann_test(softmax_derivative "binary_operators_test/test_softmax_derivative.cpp")
add_metann_test(subtration_test "binary_operators_test/test_subtraction.cpp")
add_metann_test(tanh_derivative_test "binary_operators_test/test_tanh_derivative.cpp")

### Elementary layer tests
add_metann_test(activate_layer_test "elementary_layer/test_activate.cpp")
add_metann_test(weight_layer_test "elementary_layer/test_weight_layer.cpp")
add_metann_test(filler_test "elementary_layer/test_filler.cpp")
add_metann_test(operator_layer_test "elementary_layer/test_oper_layer.cpp")
add_metann_test(bias_layer_test "elementary_layer/test_bias.cpp")

### Composed layer tests
add_metann_test(compose_kernel_test "composed_layer/test_compose_kernel.cpp")
add_metann_test(linear_layer_test "composed_layer/test_linear_layer.cpp")
add_metann_test(single_layer_test "composed_layer/test_single_layer.cpp")