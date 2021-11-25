// Can be compiled as
// g++ cuda-ctxsize.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lnvidia-ml
// Returns 305Mb for driver 450.119.03 on V100
#include <nvml.h>
#include <cuda_runtime.h>

#include <iostream>
#include <unistd.h>

unsigned long long getGPUMemory() {
  auto raiseError = [](const std::string& msg, nvmlReturn_t rc) {
    throw std::runtime_error(msg + " " +   nvmlErrorString(rc) + " (" + std::to_string(rc) + ")");
  };

  auto rc = nvmlInit();
  if (rc != NVML_SUCCESS) {
     raiseError("nvmlInit() failed", rc);
  }

  nvmlDevice_t device = nullptr;
  rc = nvmlDeviceGetHandleByIndex(0, &device);
  if (rc != NVML_SUCCESS) {
    raiseError("nvmlDeviceGetHandleByIndex() failed", rc);
  }

  unsigned int infoCount = 0;
  rc = nvmlDeviceGetComputeRunningProcesses(device, &infoCount, nullptr);
  if (rc == NVML_SUCCESS && infoCount == 0) {
	  return 0;
  }
  if (rc != NVML_ERROR_INSUFFICIENT_SIZE) {
    raiseError("nmlDeviceGetComputeRuningProcesses() failed", rc);
  }
  std::vector<nvmlProcessInfo_t> infos(infoCount);
  rc = nvmlDeviceGetComputeRunningProcesses(device, &infoCount, infos.data());
  if (rc != NVML_SUCCESS) {
    raiseError("nmlDeviceGetComputeRuningProcesses() failed", rc);
  }
  for(auto info: infos) {
    if (info.pid == getpid()) {
      return info.usedGpuMemory;
    }
  }
  return 0;
}


void printGPUMemory(const std::string& msg) {
  try {
    std::cout << msg << " " << getGPUMemory() / 1048576  << "MiB" << std::endl;
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    std::exit(-1);
  }
}

int main(void) {
	cudaFree(0);
	printGPUMemory("Total mem");
}
