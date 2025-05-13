#include <iostream>
#include "MyNN.h"

using namespace MyNN;

int main()
{
    NNLayer layer(4, 2);
    Matrix in(1, 2);

    layer.Randomize();
    randomize(in);

    layer.Calculate(in, [](float) -> float { return 0; });

    print(layer.GetOutputData(), "out");
}