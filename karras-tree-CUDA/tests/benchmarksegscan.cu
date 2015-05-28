/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 *
 * Code and text by Sean Baxter, NVIDIA Research
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 *
 ******************************************************************************/

#include "../kernels/scan.cuh"

#include "../../../moderngpu/include/util/format.h"

#include <cstdlib>

int myRand(int min, int max) {
    return rand() % (max + 1 - min) + min;
}

enum TestType {
    TestTypeNormal,
    TestTypePreprocess
};


template<typename T>
void TestCsrScan(int count, int randomSize, int numIterations,
    TestType testType, bool supportEmpty, mgpu::CudaContext& context) {

#ifdef _DEBUG
    numIterations = 1;
#endif

    std::vector<int> segCountsHost, csrHost;
    int total = 0;
    int numValidRows = 0;
    while(total < count) {
        int randMin = supportEmpty ? 0 : 1;
        int segSize = myRand(randMin, min(randomSize, count - total));
        numValidRows += 0 != segSize;
        csrHost.push_back(total ? (csrHost.back() + segCountsHost.back()) : 0);
        segCountsHost.push_back(segSize);
        total += segSize;
    }
    int numRows = (int)segCountsHost.size();

    std::vector<int> sourcesHost(numRows);
    for(int i = 0; i < numRows; ++i)
        sourcesHost[i] = myRand(0, max(0, count - randomSize));

    MGPU_MEM(int) csrDevice = context.Malloc(csrHost);
    MGPU_MEM(int) sourcesDevice = context.Malloc(sourcesHost);

    // Generate random ints as input.
    std::vector<T> dataHost(count);
    for(int i = 0; i < count; ++i)
        dataHost[i] = (T)myRand(1, 9);
    MGPU_MEM(T) dataDevice = context.Malloc(dataHost);

    MGPU_MEM(T) resultsDevice = context.Malloc<T>(count);

    std::auto_ptr<grace::SegScanPreprocessData> preprocessData;
    grace::SegScanCsrPreprocess<T>(count, csrDevice->get(), numRows,
                                   supportEmpty, &preprocessData, context);

    context.Start();
    for(int it = 0; it < numIterations; ++it) {
        if(TestTypeNormal == testType)
            mgpu::SegReduceCsr(dataDevice->get(), csrDevice->get(), count, numRows,
                supportEmpty, resultsDevice->get(), (T)0, mgpu::plus<T>(),
                context);
        else
            grace::SegScanApply(*preprocessData, dataDevice->get(), (T)0,
                                mgpu::plus<T>(), resultsDevice->get(), context);
    }
    double elapsed = context.Split();
    double throughput = (double)numIterations * count / elapsed;

    printf("%9.3lf M/s  %9.3lf GB/s\n", throughput / 1.0e6,
        sizeof(T) * throughput / 1.0e9);

    std::vector<T> resultsHost;
    resultsDevice->ToHost(resultsHost);

    std::vector<T> resultsRef(count);
    for(int row = 0; row < numRows; ++row) {
        int begin = csrHost[row];
        int end = (row + 1 < numRows) ? csrHost[row + 1] : count;

        T x = 0;
        for(int i = begin; i < end; ++i) {
            resultsRef[i] = x;
            x = x + dataHost[i];

            if(resultsRef[i] != resultsHost[i]) {
                printf("SCAN ERROR ON ELEMENT %d OF %d, IN SEGMENT %d of %d\n",
                       i - begin, end - begin - 1, row, numRows);
                printf("ELEMENT %d OF %d\n", i, count);
                printf("MGPU result %.5g\nRef. result %.5g\n",
                       resultsHost[i], resultsRef[i]);
                if (i)
                    printf("Prior MGPU result %.5g\nPrior Ref. result %.5g\n",
                           resultsHost[i-1], resultsRef[i-1]);
                if (i < count - 1)
                    printf("Next MGPU result %.5g\nNext Ref. result %.5g\n",
                           resultsHost[i+1], x);
                exit(0);
            }
        }
    }
}

const int Tests[][2] = {
    { 10000, 10000 },
    { 50000, 10000 },
    { 100000, 10000 },
    { 200000, 5000 },
    { 500000, 2000 },
    { 1000000, 2000 },
    { 2000000, 2000 },
    { 5000000, 2000 },
    { 10000000, 1000 },
    { 20000000, 1000 },
    { 60000000, 200 },
    {120000000, 100 },
    {200000000, 50 },
};
const int NumTests = sizeof(Tests) / sizeof(*Tests);

const int SegSizes[] = {
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
    500000
};
const int NumSegSizes = sizeof(SegSizes) / sizeof(*SegSizes);

template<typename T>
void BenchmarkSegScan1(TestType testType, bool supportEmpty,
    mgpu::CudaContext& context) {
    int avSegSize = 500;

    const char* typeString;
    if(TestTypeNormal == testType) typeString = "seg";
    else typeString = "preprocess";

    printf("Benchmarking %s-scan type %s. AvSegSize = %d.\n",
        typeString, mgpu::TypeIdName<T>(), avSegSize);

    for(int test = 0; test < NumTests; ++test) {
        int count = Tests[test][0];

        printf("%8s: ", mgpu::FormatInteger(count).c_str());
        TestCsrScan<T>(count, 2 * avSegSize, Tests[test][1], testType,
            supportEmpty, context);

        context.GetAllocator()->Clear();
    }
    printf("\n");
}

template<typename T>
void BenchmarkSegScan2(TestType testType, bool supportEmpty,
    mgpu::CudaContext& context) {

    int count = 20000000;
    int numIterations = 500;

    const char* typeString;
    if(TestTypeNormal == testType) typeString = "seg";
    else typeString = "preprocess";

    printf("Benchmarking %s-scan type %s. Count = %d.\n",
        typeString, mgpu::TypeIdName<T>(), count);

    for(int test = 0; test < NumSegSizes; ++test) {
        int avSegSize = SegSizes[test];

        printf("%8s: ", mgpu::FormatInteger(avSegSize).c_str());
        TestCsrScan<T>(count, 2 * avSegSize, numIterations, testType,
            supportEmpty, context);

        context.GetAllocator()->Clear();
    }
    printf("\n");
}

int main(int argc, char** argv) {
    mgpu::ContextPtr context = mgpu::CreateCudaDevice(argc, argv, true);

    bool supportEmpty = false;
    TestType testType = TestTypePreprocess;

    BenchmarkSegScan1<float>(testType, supportEmpty, *context);
    BenchmarkSegScan1<double>(testType, supportEmpty, *context);

    BenchmarkSegScan2<float>(testType, supportEmpty, *context);
    BenchmarkSegScan2<double>(testType, supportEmpty,  *context);

    return 0;
}
