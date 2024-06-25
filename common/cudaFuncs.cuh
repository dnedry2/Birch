#ifndef __CUDA_FUNCS_CUH__
#define __CUDA_FUNCS_CUH__

int  cuda_count_devices();
void cuda_list_devices();

void cuda_get_device_memory(int device, size_t *free, size_t *total);

void cuda_init();

#endif