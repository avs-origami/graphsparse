#include <format>
#include <iostream>
#include <sstream>
#include <vector>

#include <torch/torch.h>
#include <torch/extension.h>

using Tensor = torch::Tensor;

struct Color {
    int r, g, b; 
};

// const std::vector<Color> THEME {
//     {0, 0, 0},
//     {32, 6, 80},
//     {64, 17, 159},
//     {255, 55, 140},
//     {255, 240, 180},
//     {255, 255, 255},  
// };

const std::vector<Color> THEME {
    {0, 0, 255},
    {0, 0, 0},
    {255, 0, 0},
};

Tensor map_range(Tensor x, float min, float max) {
    // Clamp values to the specified range
    float mi = -3.0;
    float ma = 3.0;
    auto clamped = torch::clamp(x, mi, ma);
    // Map to [0, 1] using the fixed range
    auto mapped = (clamped - mi) / (ma - mi);
    return mapped;

    // auto compressed = torch::tanh(x / 3.0);
    
    // // Map from [-1, 1] to [0, 1]
    // auto mapped = (compressed + 1.0) / 2.0;
    // return mapped;
}

Color interpolateColor(const Color &color1, const Color &color2, double t) {
    Color result;
    result.r = color1.r + (color2.r - color1.r) * t;
    result.g = color1.g + (color2.g - color1.g) * t;
    result.b = color1.b + (color2.b - color1.b) * t;
    return result;
}

Color mapValueToColor(double value, const std::vector<Color> &colorGradient) {
    // Clamp value between 0 and 1
    if (value <= 0.0) return colorGradient[0];
    if (value >= 1.0) return colorGradient.back();

    // Calculate position in gradient
    double scaledPosition = value * (colorGradient.size() - 1);
    size_t index = static_cast<size_t>(scaledPosition);
    double t = scaledPosition - index;

    // If we're at the last color, return it
    if (index >= colorGradient.size() - 1) {
        return colorGradient.back();
    }

    return interpolateColor(colorGradient[index], colorGradient[index + 1], t);
}

// Function to print colored text using RGB values
void printColored(const std::string &text, const Color &fg, const std::optional<Color> &bg = std::nullopt) {
    std::cerr << "\033[38;2;" << fg.r << ";" << fg.g << ";" << fg.b << "m";  // Change the foreground color

    if (bg.has_value()) {
        auto bgc = bg.value();
        std::cerr << "\033[48;2;" << bgc.r << ";" << bgc.g << ";" << bgc.b << "m";  // Change the background color
    }

    std::cerr << text << "\033[0m";  // Display the text and reset formatting
}

// Function to print a tensor via colored boxes
void printTensor(const Tensor &x) {
    auto tensor = x.clone();
    auto sizes = tensor.sizes();
    if (sizes.size() > 2) {
        std::ostringstream oss;
        oss << std::vector<int>(sizes.begin(), sizes.end());
        std::string vec_str = oss.str();
        throw std::invalid_argument(std::format("printTensor: tensor can be at most 2D! got shape [{}]", vec_str));
    } else if (sizes.size() == 1) {
        tensor = tensor.unsqueeze(1);
    }

    auto m = map_range(tensor, 0.0, 1.0);

    for (int i = 0; i < sizes[0]; i += 2) {
        for (int j = 0; j < sizes[1]; j ++) {
            if (i + 1 < sizes[0]) {
                auto topval = m[i][j].item<float>();
                auto botval = m[i + 1][j].item<float>();
                printColored("▀", mapValueToColor(topval, THEME), mapValueToColor(botval, THEME));
            } else {
                auto topval = m[i][j].item<float>();
                printColored("▀", mapValueToColor(topval, THEME));
            }
        }

        std::cerr << std::endl;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tviz", &printTensor, "Visualize tensor in color");
}