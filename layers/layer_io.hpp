//
// Created by asus on 2025/1/14.
//

#ifndef LAYER_IO_HPP
#define LAYER_IO_HPP
#include "../utils/vartype_dict.hpp"

namespace metann
{
    struct LayerIO : public VarTypeDict<LayerIO>
    {
    };

    struct CostLayerIO : public VarTypeDict<CostLayerIO, struct CostLayerLabel1>
    {
    };
}

#endif //LAYER_IO_HPP
