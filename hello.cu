#include <iostream>
#include <set>

#include <stdio.h>

__global__ void kernel() {
  printf("Hello World of CUDA %d\n", threadIdx.x);
}

void exitIfFailed(const std::string& msg, cudaError_t err)
{
  if (err == cudaSuccess) {
    return;
  }
  std::cerr << msg << " " << cudaGetErrorString(err) << " (" << err << ")" << std::endl;
  std::exit(-1);
}

void detectCompileFlags() {
  int deviceCount = -1;
  auto rc = cudaGetDeviceCount(&deviceCount);
  exitIfFailed("cudaGetDeviceCount() failed", rc);
  std::set<int> sms;
  std::cout << "GPUs present on the system" << std::endl;
  for(int cnt=0; cnt < deviceCount; ++cnt) {
    cudaDeviceProp prop;
    rc = cudaGetDeviceProperties(&prop, cnt);
          exitIfFailed("cudaGetDeviceProp() failed", rc);
    sms.insert(prop.major*10 + prop.minor);
    std::cout << "  " << cnt << " : " << prop.name << " ComputeCapabilities " << prop.major << "." << prop.minor << std::endl;
  }
        std::cout << "For optimal performance/binary forward compatibiliy,"
            << " please make sure code is compiled with the follwing options"
      << std::endl;
  for(auto sm: sms) {
    std::cout << "  -gencode arch=compute_" << sm << ",code=sm_" << sm << std::endl;
  }
}

int main() {
  // cudaFree is a no op that initializes GPU context and also the best way to check for errors
  auto rc = cudaFree(0);
  if (rc != cudaSuccess) {
    std::cout << "Creating cudaRuntime context failed with " << cudaGetErrorString(rc) << " (" << rc << ")" << std::endl;
    detectCompileFlags();
    std::exit(-1);
  }
  kernel<<<1,1>>>();
  // Checking error code here is the only way of knowing if kernel launch failed for some reason
  rc = cudaGetLastError();
  exitIfFailed("Launch failed", rc);
  rc = cudaDeviceSynchronize();
  exitIfFailed("Kernel execution failed", rc);
  return 0;
}
