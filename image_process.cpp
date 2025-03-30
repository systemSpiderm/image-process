#include <mpi.h>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <immintrin.h>
#include <omp.h>

using std::vector;
#define THREAD_COUNT 8

#define idx(i,j) ((i)*16384 + (j))

unsigned char* figure;
unsigned char* result;

void initialize(size_t N, size_t seed = 0);

void gaussianFilter(unsigned char* fig, unsigned char* res, int rows_per_proc, size_t N, 
                    int rank, int size, MPI_Comm comm);

void powerLawTransformation(unsigned char* fig, unsigned char* res, int rows_per_proc, size_t N, unsigned char* LUT);

void runBenchmark(int rank, int size, size_t N);


unsigned int calcChecksum(int N) {
    unsigned int sum = 0;
    constexpr size_t mod = 1000000007;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            sum += result[i*N + j];
            sum %= mod;
        }
    }
    return sum;
}


// Main function
int main(int argc, char* argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    constexpr size_t N = 16384;
    if (rank == 0)
        initialize(N, argc > 1 ? std::stoul(argv[1]) : 0);

    runBenchmark(rank, size, N);

    delete[] figure;
    delete[] result;

    MPI_Finalize();
    return 0;
}


void initialize(size_t N, size_t seed) {
    figure = new unsigned char[N * N];
    result = new unsigned char[N * N]();
    // !!! Please do not modify the following code !!!
    std::random_device rd;
    std::mt19937_64 gen(seed == 0 ? rd() : seed);
    std::uniform_int_distribution<unsigned char> distribution(0, 255);
    // !!! ----------------------------------------- !!!

    // 数组初始化
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            figure[i*N + j] = static_cast<unsigned char>(distribution(gen));
        }
    }
}

void gaussianFilter(unsigned char* fig, unsigned char* res, int rows_per_proc, size_t N, 
                    int rank, int size, MPI_Comm comm) {
    // 与上/下邻居交换边界行
    if (rank > 0) 
        MPI_Sendrecv(&fig[N], N, MPI_UNSIGNED_CHAR, rank - 1, 0, &fig[0], N, MPI_UNSIGNED_CHAR, 
                        rank - 1, 0, comm, MPI_STATUS_IGNORE);

    if (rank < size - 1) 
        MPI_Sendrecv(&fig[rows_per_proc * N], N, MPI_UNSIGNED_CHAR, rank + 1, 0, &fig[(rows_per_proc + 1) * N], 
                        N, MPI_UNSIGNED_CHAR, rank + 1, 0, comm, MPI_STATUS_IGNORE);


    #pragma omp parallel for num_threads(THREAD_COUNT)
    for (int i = 1; i < rows_per_proc + 1; i++) {
        // 首行迭代（包括左上和右上）
        if (i == 1 && rank == 0) {
            res[1*N + 0] = (4*fig[1*N + 0] + 2*fig[1*N + 1] + 2*fig[2*N + 0] + fig[2*N + 1]) / 9.0;
            res[1*N + N-1] = (4*fig[1*N + N-1] + 2*fig[1*N + N-2] + 2*fig[2*N + N-1] + fig[2*N + N-2]) / 9.0;
            for (int j = 1; j < N - 1; j++) {
                res[i*N + j] = (fig[i*N + j-1] + 2*fig[i*N + j] + fig[i*N + j+1] + 
                            2*fig[i*N + j-1] + 4*fig[i*N + j] + 2*fig[i*N + j+1] + 
                            fig[(i+1)*N + j-1] + 2*fig[(i+1)*N + j] + fig[(i+1)*N + j+1]) / 16.0;
            }
            continue;
        }
        // 尾行迭代（包括左下和右下）
        if (i == rows_per_proc && rank == size - 1) {
            res[i*N + 0] = (4*fig[i*N + 0] + 2*fig[i*N + 1] + 2*fig[(i-1)*N + 0] + fig[(i-1)*N + 1]) / 9.0;
            res[i*N + N-1] = (4*fig[i*N + N-1] + 2*fig[i*N + N-2] + 2*fig[(i-1)*N + N-1] + fig[(i-1)*N + N-2]) / 9.0;
            for (int j = 1; j < N - 1; j++) {
                res[i*N + j] = (fig[(i-1)*N + j-1] + 2*fig[(i-1)*N + j] + fig[(i-1)*N + j+1] + 
                            2*fig[i*N + j-1] + 4*fig[i*N + j] + 2*fig[i*N + j+1] + 
                            fig[i*N + j-1] + 2*fig[i*N + j] + fig[i*N + j+1]) / 16.0;
            }
            continue;
        }

        // 首列迭代
        res[i*N + 0] = (fig[(i-1)*N + 0] + 2*fig[(i-1)*N + 0] + fig[(i-1)*N + 1] + 
                        2*fig[i*N + 0] + 4*fig[i*N + 0] + 2*fig[i*N + 1] + 
                        fig[(i+1)*N + 0] + 2*fig[(i+1)*N + 0] + fig[(i+1)*N + 1]) / 16.0;

        // 尾列迭代
        res[i*N + N-1] = (fig[(i-1)*N + N-2] + 2*fig[(i-1)*N + N-1] + fig[(i-1)*N + N-1] + 
                        2*fig[i*N + N-2] + 4*fig[i*N + N-1] + 2*fig[i*N + N-1] + 
                        fig[(i+1)*N + N-2] + 2*fig[(i+1)*N + N-1] + fig[(i+1)*N + N-1]) / 16.0;
            
        // 内部迭代
        for (int j = 1; j <= N - 32 - 1; j += 32) {
            
            __m256i top_left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&fig[(i-1)*N + (j-1)]));
            __m256i top = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&fig[(i-1)*N + j]));
            __m256i top_right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&fig[(i-1)*N + (j+1)]));
            __m256i mid_left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&fig[i*N + (j-1)]));
            __m256i mid = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&fig[i*N + j]));
            __m256i mid_right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&fig[i*N + (j+1)]));
            __m256i bottom_left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&fig[(i+1)*N + (j-1)]));
            __m256i bottom = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&fig[(i+1)*N + j]));
            __m256i bottom_right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&fig[(i+1)*N + (j+1)]));


            __m256i top_left_1 = _mm256_unpacklo_epi8(top_left, _mm256_setzero_si256());
            __m256i top_left_2 = _mm256_unpackhi_epi8(top_left, _mm256_setzero_si256());
            __m256i top_1 = _mm256_unpacklo_epi8(top, _mm256_setzero_si256());
            __m256i top_2 = _mm256_unpackhi_epi8(top, _mm256_setzero_si256());
            __m256i top_right_1 = _mm256_unpacklo_epi8(top_right, _mm256_setzero_si256());
            __m256i top_right_2 = _mm256_unpackhi_epi8(top_right, _mm256_setzero_si256());
            __m256i mid_left_1 = _mm256_unpacklo_epi8(mid_left, _mm256_setzero_si256());
            __m256i mid_left_2 = _mm256_unpackhi_epi8(mid_left, _mm256_setzero_si256());
            __m256i mid_1 = _mm256_unpacklo_epi8(mid, _mm256_setzero_si256());
            __m256i mid_2 = _mm256_unpackhi_epi8(mid, _mm256_setzero_si256());
            __m256i mid_right_1 = _mm256_unpacklo_epi8(mid_right, _mm256_setzero_si256());
            __m256i mid_right_2 = _mm256_unpackhi_epi8(mid_right, _mm256_setzero_si256());
            __m256i bottom_left_1 = _mm256_unpacklo_epi8(bottom_left, _mm256_setzero_si256());
            __m256i bottom_left_2 = _mm256_unpackhi_epi8(bottom_left, _mm256_setzero_si256());
            __m256i bottom_1 = _mm256_unpacklo_epi8(bottom, _mm256_setzero_si256());
            __m256i bottom_2 = _mm256_unpackhi_epi8(bottom, _mm256_setzero_si256());
            __m256i bottom_right_1 = _mm256_unpacklo_epi8(bottom_right, _mm256_setzero_si256());
            __m256i bottom_right_2 = _mm256_unpackhi_epi8(bottom_right, _mm256_setzero_si256());

            __m256i sum_1 = _mm256_setzero_si256();
            sum_1 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16
                                    (_mm256_add_epi16(_mm256_add_epi16(_mm256_setzero_si256(), top_left_1), _mm256_slli_epi16(top_1, 1)), top_right_1), 
                                    _mm256_slli_epi16(mid_left_1, 1)),_mm256_slli_epi16(mid_1, 2)), _mm256_slli_epi16(mid_right_1, 1)), bottom_left_1), 
                                    _mm256_slli_epi16(bottom_1, 1)), bottom_right_1), 4);

            __m256i sum_2 = _mm256_setzero_si256();
            _mm256_storeu_si256((__m256i*)(&res[idx(i, j)]), _mm256_packus_epi16(sum_1, sum_2));

        }

        // 处理剩余的像素
        for (int j = (N - 2) / 32 * 32 + 1; j < N - 1; j++) {
            res[i*N + j] = (fig[(i-1)*N + j-1] + 2*fig[(i-1)*N + j] + fig[(i-1)*N + j+1] + 
                            2*fig[i*N + j-1] + 4*fig[i*N + j] + 2*fig[i*N + j+1] +
                            fig[(i+1)*N + j-1] + 2*fig[(i+1)*N + j] + fig[(i+1)*N + j+1]) / 16;
        }
    }
}

void powerLawTransformation(unsigned char* fig, unsigned char* res, int rows_per_proc, size_t N, unsigned char* LUT) {
    // 每次处理32个像素（256位AVX寄存器）

    
    #pragma omp parallel for num_threads(THREAD_COUNT)
    for (int i = 1; i < rows_per_proc + 1; i++) {
        for (int j = 0; j < N; j += 32) {  // 每次处理32个像素
            // 加载32个像素值到AVX寄存器中
            __m256i pixels = _mm256_loadu_si256((__m256i*)&fig[i*N + j]);
            
            // 从LUT加载相应的转换值
            __m256i lut_values = _mm256_set_epi8(
                LUT[fig[i*N + j+31]], LUT[fig[i*N + j+30]], LUT[fig[i*N + j+29]], LUT[fig[i*N + j+28]],
                LUT[fig[i*N + j+27]], LUT[fig[i*N + j+26]], LUT[fig[i*N + j+25]], LUT[fig[i*N + j+24]],
                LUT[fig[i*N + j+23]], LUT[fig[i*N + j+22]], LUT[fig[i*N + j+21]], LUT[fig[i*N + j+20]],
                LUT[fig[i*N + j+19]], LUT[fig[i*N + j+18]], LUT[fig[i*N + j+17]], LUT[fig[i*N + j+16]],
                LUT[fig[i*N + j+15]], LUT[fig[i*N + j+14]], LUT[fig[i*N + j+13]], LUT[fig[i*N + j+12]],
                LUT[fig[i*N + j+11]], LUT[fig[i*N + j+10]], LUT[fig[i*N + j+9]], LUT[fig[i*N + j+8]],
                LUT[fig[i*N + j+7]], LUT[fig[i*N + j+6]], LUT[fig[i*N + j+5]], LUT[fig[i*N + j+4]],
                LUT[fig[i*N + j+3]], LUT[fig[i*N + j+2]], LUT[fig[i*N + j+1]], LUT[fig[i*N + j]]
            );
            
            // 将转换结果存储到结果数组中
            _mm256_storeu_si256((__m256i*)&res[i*N + j], lut_values);
        }
    }
}


    

void runBenchmark(int rank, int size, size_t N) {

    int rows_per_proc;
    unsigned char *local_fig, *local_res;
    unsigned int sum;
    rows_per_proc = N / size;


    // 提前构造查找表（向量版本）
    unsigned char LUT[256] = {0};
    constexpr float gamma = 0.5f;
    for (int i = 1; i < 256; i++) 
        LUT[i] = static_cast<unsigned char>(255.0f * std::pow(i / 255.0f, gamma) + 0.5f);
    

    // 为每个进程分配局部矩阵（包含上下邻居的额外一行）
    local_fig = new unsigned char[(rows_per_proc + 2) * N * sizeof(unsigned char)];
    local_res = new unsigned char[(rows_per_proc + 2) * N * sizeof(unsigned char)]();

    // 广播初始条件
    MPI_Scatter(figure, rows_per_proc * N, MPI_UNSIGNED_CHAR, 
                &local_fig[N], rows_per_proc * N, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    gaussianFilter(local_fig, local_res, rows_per_proc, N, rank, size, MPI_COMM_WORLD);

    // MPI归约
    MPI_Gather(&local_res[N], rows_per_proc * N, MPI_UNSIGNED_CHAR, result, 
                rows_per_proc * N, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
                
    MPI_Barrier(MPI_COMM_WORLD);
    auto middle = std::chrono::high_resolution_clock::now();

    if (rank == 0) 
        sum = calcChecksum(N);


    
    MPI_Barrier(MPI_COMM_WORLD);
    auto middle2 = std::chrono::high_resolution_clock::now();

    powerLawTransformation(local_fig, local_res, rows_per_proc, N, LUT);

    // MPI归约
    MPI_Gather(&local_res[N], rows_per_proc * N, MPI_UNSIGNED_CHAR, result, rows_per_proc * N, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    delete[] local_fig;
    delete[] local_res;

    if (rank == 0) {
        sum += calcChecksum(N);
        sum %= 1000000007;
        std::cout << "Checksum: " << sum << "\n";

        auto milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(middle - start) +
            std::chrono::duration_cast<std::chrono::milliseconds>(end - middle2);
        std::cout << "Benchmark time: " << milliseconds.count() << " ms\n";
    }
}