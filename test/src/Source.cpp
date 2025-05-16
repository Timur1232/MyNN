#include <iostream>
#include "MyNN.h"
#include <cmath>
#include <format>

using namespace MyNN;

double sigmoid(double x)
{
    return 1.0f / (1.0f + std::expf(-x));
}

void forward_print(NeuralNetwork& nn, Matrix& trainData)
{
    for (size_t row = 0; row < 4; ++row)
    {
        const auto& input = trainData.GetRow(row);
        nn.PropagateForward(input);
        double answer = nn.GetOutput().At(0, 0);
        std::cout << std::format("{:.0f} ^ {:.0f} = {:.3f}\n",
            input[0], input[1], answer);
    }
    std::cout << '\n';
}

int main()
{
    std::srand(std::time(0));
    NeuralNetwork nn({ 2, 3, 1 }, sigmoid);
    nn.Randomize();
    nn.PrintInfo();

    Matrix trainData(4, 2);
    trainData.At(0, 0) = 0;
    trainData.At(1, 0) = 0;
    trainData.At(2, 0) = 1;
    trainData.At(3, 0) = 1;

    trainData.At(0, 1) = 0;
    trainData.At(1, 1) = 1;
    trainData.At(2, 1) = 0;
    trainData.At(3, 1) = 1;

    PRINT_MATRIX(trainData);
    std::cout << '\n';

    data_vector desired = {0, 1, 1, 0};

    forward_print(nn, trainData);
    std::cout << "cost = " << nn.CalculateCost(trainData, desired) << '\n';

    NeuralNetwork grad({ 2, 3, 1 }, sigmoid);
    for (size_t i = 0; i < 10000; ++i)
    {
        nn.ForwardDifference(trainData, desired, grad);
        //std::cout << "cost = " << nn.CalculateCost(trainData, desired) << '\n';
    }
    std::cout << "cost = " << nn.CalculateCost(trainData, desired) << "\n\n";

    forward_print(nn, trainData);
}