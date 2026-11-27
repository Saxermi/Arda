#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Clock = std::chrono::steady_clock;

bool read_tensor(const char *path, Matrix &matrix) {
    std::ifstream input(path, std::ios::binary);
    input.seekg(128);
    input.read(reinterpret_cast<char *>(matrix.data()), matrix.size() * sizeof(double));
    return bool(input);
}

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.'
                  << EIGEN_MINOR_VERSION << '\n';
        return 0;
    }
    if (argc != 4)
        return 2;
    const int n = std::stoi(argv[1]), rounds = std::stoi(argv[2]), repeat = std::stoi(argv[3]);
    Eigen::setNbThreads(1);
    Matrix a(n, n), b(n, n);
    if (!read_tensor("a.atn", a) || !read_tensor("b.atn", b))
        return 1;
    Matrix result = a * b;
    std::cout << std::setprecision(17);
    for (int round = 0; round < rounds; round++) {
        double elapsed = 0;
        for (int i = 0; i < repeat; i++) {
            const auto start = Clock::now();
            Matrix next = a * b;
            result.swap(next);
            elapsed += std::chrono::duration<double>(Clock::now() - start).count();
        }
        std::cout << elapsed / repeat << '\n';
    }
    std::ofstream output("result.raw", std::ios::binary);
    output.write(reinterpret_cast<const char *>(result.data()), result.size() * sizeof(double));
    return output ? 0 : 1;
}
