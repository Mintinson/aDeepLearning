add_executable(chapter6 chapter6.cpp
        ../layers/policies/init_policy.hpp
        ../layers/initializer.hpp
        ../layers/dynamic_data.hpp
        ../layers/policies/update_policy.hpp
        ../layers/policies/input_policy.hpp
        ../layers/policies/operand_policy.hpp
        ../layers/elementary/add_layer.hpp
        ../layers/layer_io.hpp
        ../layers/elementary/mul_layer.hpp
        ../layers/layer_helper.hpp
        ../layers/elementary/bias_layer.hpp
        ../layers/elementary/sigmoid_layer.hpp
        ../layers/elementary/softmax_layer.hpp
        ../layers/elementary/tanh_layer.hpp
        ../layers/elementary/abs_layer.hpp
        ../layers/elementary/weight_layer.hpp
        ../layers/fillers/var_scale_filler.hpp)
        
add_executable(chapter6_1 chapter6_1.cpp)
add_executable(chapter6_2 chapter6_2.cpp "../policy/policy.hpp")