function(add_metann_test TARGET_NAME SOURCE_FILE)
	add_executable(${TARGET_NAME} ${SOURCE_FILE})
	target_link_libraries(${TARGET_NAME} PRIVATE metann)
	add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endfunction()

set(METANN_TEST_CASES
	"utils_test|utils_test/test_utils.cpp"

	"array_test|datatype_test/test_arr.cpp"
	"matrix_test|datatype_test/test_matrix.cpp"
	"dup_batch_test|datatype_test/test_batch_duplicate.cpp"
	"other_data_test|datatype_test/test_others.cpp"

	"abs_test|unary_operators_test/test_abs.cpp"
	"transpose_test|unary_operators_test/test_transpose.cpp"
	"tanh_test|unary_operators_test/test_tanh.cpp"
	"sign_test|unary_operators_test/test_sign.cpp"
	"softmax_test|unary_operators_test/test_softmax.cpp"
	"sigmoid_test|unary_operators_test/test_sigmoid.cpp"
	"collapse_test|unary_operators_test/test_collapse.cpp"

	"dot_test|binary_operators_test/test_dot.cpp"
	"add_test|binary_operators_test/test_add.cpp"
	"divide_test|binary_operators_test/test_divide.cpp"
	"mul_test|binary_operators_test/test_mul.cpp"
	"sigmoid_derivative|binary_operators_test/test_sigmoid_derivative.cpp"
	"softmax_derivative|binary_operators_test/test_softmax_derivative.cpp"
	"subtration_test|binary_operators_test/test_subtraction.cpp"
	"tanh_derivative_test|binary_operators_test/test_tanh_derivative.cpp"

	"activate_layer_test|elementary_layer/test_activate.cpp"
	"weight_layer_test|elementary_layer/test_weight_layer.cpp"
	"filler_test|elementary_layer/test_filler.cpp"
	"operator_layer_test|elementary_layer/test_oper_layer.cpp"
	"bias_layer_test|elementary_layer/test_bias.cpp"

	"compose_kernel_test|composed_layer/test_compose_kernel.cpp"
	"linear_layer_test|composed_layer/test_linear_layer.cpp"
	"single_layer_test|composed_layer/test_single_layer.cpp"
)

foreach(test_case IN LISTS METANN_TEST_CASES)
	string(REPLACE "|" ";" test_case_parts "${test_case}")
	list(GET test_case_parts 0 target_name)
	list(GET test_case_parts 1 source_file)
	add_metann_test(${target_name} ${source_file})
endforeach()

if(METANN_BUILD_EXPERIMENTAL_TESTS)
	add_metann_test(cost_layer_test "cost_layer_test/test_negative_log_likelihood_layer.cpp")
endif()