#include <iostream>
#include "MyNN.h"
#include <cmath>

using namespace MyNN;

float sigmoid(float x)
{
    return 1.0f / (1.0f + std::expf(-x));
}

int main()
{
    NNLayer layer(4, 2);
    Matrix in(1, 2);

    layer.Randomize(0.0f, 10.0f);
    randomize(in, 0.0f, 10.0f);

    layer.Forward(in, sigmoid);

    print(layer.GetOutputData(), "out");
}