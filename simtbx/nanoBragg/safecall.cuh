

#ifndef SIMTBX_NANOBRAGG_SAFECALL_CUH
#define SIMTBX_NANOBRAGG_SAFECALL_CUH

//https://stackoverflow.com/a/14038590/2077270
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline void error_msg(cudaError_t err, const char* msg){
    if (err != cudaSuccess){
        printf("%s: CUDA error message: %s\n", msg, cudaGetErrorString(err));
        exit(err);
    }
}
#endif // SIMTBX_NANOBRAGG_SAFECALL_CUH
