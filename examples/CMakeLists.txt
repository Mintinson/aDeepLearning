function(add_metann_example TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PRIVATE metann)
endfunction()

set(METANN_EXAMPLE_CASES
    "ml_linear_regression|ml_linear_regression.cpp"
    "ml_logistic_regression|ml_logistic_regression.cpp"
    "nn_two_layer_inference|nn_two_layer_inference.cpp"
    "nn_single_layer_backward|nn_single_layer_backward.cpp"
    "nn_xor_training|nn_xor_training.cpp"
    "nn_mnist_training|nn_mnist_training.cpp"
)

foreach(example_case IN LISTS METANN_EXAMPLE_CASES)
    string(REPLACE "|" ";" example_case_parts "${example_case}")
    list(GET example_case_parts 0 target_name)
    list(GET example_case_parts 1 source_file)
    add_metann_example(${target_name} ${source_file})
endforeach()
