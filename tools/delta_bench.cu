// delta_bench -- is k_delta_step (the GDN recurrence, P0: 2.52 ms = biggest
// non-weight kernel in a wide round) bandwidth-bound, occupancy-bound, or
// write-redundant? Measures the shipped kernel in isolation + a SOL state-copy
// (the 6 MB read+write floor) + a register-fused variant that skips the
// redundant So write/read, + a column-split variant (more blocks).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#define CK(x) do{cudaError_t e=(x);if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

static constexpr int H = 48, SK = 128;   // GDN_HEADS, GDN_DIM
static constexpr size_t STATE = (size_t)H * SK * SK; // 3 MB / 4 = floats

// ---- shipped kernel, copied verbatim from src/blocks.cu ----
__global__ void k_delta_ship(const float* __restrict__ Ssrc, float* __restrict__ Sdst,
                             const float* __restrict__ conv, const float* __restrict__ g,
                             const float* __restrict__ beta, float* __restrict__ o) {
    const int h = blockIdx.x, j = threadIdx.x & (SK-1), it = threadIdx.x >> 7, i0 = it*32;
    const int qk = h / 3;
    __shared__ float sq[SK], sk[SK], part[4][SK], dj[SK];
    const float scale = rsqrtf((float)SK);
    if (it==0){ sq[j]=conv[qk*SK+j]*scale; sk[j]=conv[2048+qk*SK+j]; }
    __syncthreads();
    const float decay = expf(g[h]);
    const float* Si = Ssrc + (size_t)h*SK*SK; float* So = Sdst + (size_t)h*SK*SK;
    float pred=0.f;
#pragma unroll 8
    for(int i=i0;i<i0+32;i++){ float s=Si[i*SK+j]*decay; So[i*SK+j]=s; pred+=sk[i]*s; }
    part[it][j]=pred; __syncthreads();
    if(it==0){ float p=part[0][j]+part[1][j]+part[2][j]+part[3][j]; float vj=conv[4096+h*SK+j]; dj[j]=beta[h]*(vj-p); }
    __syncthreads();
    float d=dj[j], acc=0.f;
#pragma unroll 8
    for(int i=i0;i<i0+32;i++){ float s=So[i*SK+j]+sk[i]*d; So[i*SK+j]=s; acc+=sq[i]*s; }
    part[it][j]=acc; __syncthreads();
    if(it==0) o[h*SK+j]=part[0][j]+part[1][j]+part[2][j]+part[3][j];
}

// ---- register-fused: hold the 32 decayed s-values in registers; write So ONCE.
// Same arithmetic -> bitwise identical (fp32 register == fp32 global). Traffic
// 12 MB -> 6 MB (drops the loop-1 So write + the loop-2 So read).
__global__ void k_delta_reg(const float* __restrict__ Ssrc, float* __restrict__ Sdst,
                            const float* __restrict__ conv, const float* __restrict__ g,
                            const float* __restrict__ beta, float* __restrict__ o) {
    const int h = blockIdx.x, j = threadIdx.x & (SK-1), it = threadIdx.x >> 7, i0 = it*32;
    const int qk = h / 3;
    __shared__ float sq[SK], sk[SK], part[4][SK], dj[SK];
    const float scale = rsqrtf((float)SK);
    if (it==0){ sq[j]=conv[qk*SK+j]*scale; sk[j]=conv[2048+qk*SK+j]; }
    __syncthreads();
    const float decay = expf(g[h]);
    const float* Si = Ssrc + (size_t)h*SK*SK; float* So = Sdst + (size_t)h*SK*SK;
    float sreg[32]; float pred=0.f;
#pragma unroll
    for(int k=0;k<32;k++){ int i=i0+k; float s=Si[i*SK+j]*decay; sreg[k]=s; pred+=sk[i]*s; }
    part[it][j]=pred; __syncthreads();
    if(it==0){ float p=part[0][j]+part[1][j]+part[2][j]+part[3][j]; float vj=conv[4096+h*SK+j]; dj[j]=beta[h]*(vj-p); }
    __syncthreads();
    float d=dj[j], acc=0.f;
#pragma unroll
    for(int k=0;k<32;k++){ int i=i0+k; float s=sreg[k]+sk[i]*d; So[i*SK+j]=s; acc+=sq[i]*s; }
    part[it][j]=acc; __syncthreads();
    if(it==0) o[h*SK+j]=part[0][j]+part[1][j]+part[2][j]+part[3][j];
}

// ---- SOL: pure read-state + write-state (the 6 MB floor a recurrence cannot beat)
__global__ void k_sol(const float* __restrict__ Si, float* __restrict__ So) {
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    for(; i<STATE; i += (size_t)gridDim.x*blockDim.x) So[i] = Si[i]*1.0000001f;
}

template<typename F> double timeit(F&& f, int reps){
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a)); for(int r=0;r<reps;r++) f(); CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms,a,b)); return ms/reps;
}

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s  %d SMs\n", p.name, p.multiProcessorCount);
    float *Si,*So,*So2,*conv,*g,*beta,*o;
    CK(cudaMalloc(&Si, STATE*4)); CK(cudaMalloc(&So, STATE*4)); CK(cudaMalloc(&So2, STATE*4));
    CK(cudaMalloc(&conv, (size_t)(4096+H*SK)*4)); CK(cudaMalloc(&g,H*4)); CK(cudaMalloc(&beta,H*4)); CK(cudaMalloc(&o,H*SK*4));
    std::vector<float> hs(STATE); for(size_t i=0;i<STATE;i++) hs[i]=sinf(0.001f*i)*0.1f;
    CK(cudaMemcpy(Si,hs.data(),STATE*4,cudaMemcpyHostToDevice));
    std::vector<float> hc(4096+H*SK); for(size_t i=0;i<hc.size();i++) hc[i]=cosf(0.002f*i)*0.2f;
    CK(cudaMemcpy(conv,hc.data(),hc.size()*4,cudaMemcpyHostToDevice));
    std::vector<float> hg(H,-0.05f), hb(H,0.7f);
    CK(cudaMemcpy(g,hg.data(),H*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(beta,hb.data(),H*4,cudaMemcpyHostToDevice));

    const double MB6 = 6.0*STATE*4/1e9;   // read Si + write So (floor), GB
    double sol = timeit([&]{ k_sol<<<p.multiProcessorCount*4,256>>>(Si,So); },200);
    printf("SOL state read+write (6MB): %.4f ms  %.0f GB/s\n", sol, 2.0*STATE*4/1e9/(sol/1e3));

    // shipped (writes So into Si-copy each time; use a fresh dst so it's not degenerate)
    double t_ship = timeit([&]{ CK(cudaMemcpy(So,Si,STATE*4,cudaMemcpyDeviceToDevice)); k_delta_ship<<<H,512>>>(Si,So,conv,g,beta,o); },200);
    // subtract the memcpy we added for a clean state; measure the kernel alone via in-place
    double t_ship2 = timeit([&]{ k_delta_ship<<<H,512>>>(Si,So2,conv,g,beta,o); },200);
    double t_reg  = timeit([&]{ k_delta_reg<<<H,512>>>(Si,So2,conv,g,beta,o); },200);
    printf("k_delta_ship (48 blk):  %.4f ms  (~%.0f GB/s at 12MB)\n", t_ship2, 12.0*STATE*4/1e9/(t_ship2/1e3)/4*4);
    printf("k_delta_reg  (48 blk):  %.4f ms  (%.1f%% of ship)\n", t_reg, 100*t_reg/t_ship2);

    // bitwise check: reg vs ship must produce identical So and o
    float *oa,*ob; CK(cudaMalloc(&oa,H*SK*4)); CK(cudaMalloc(&ob,H*SK*4));
    float *Sa,*Sb; CK(cudaMalloc(&Sa,STATE*4)); CK(cudaMalloc(&Sb,STATE*4));
    k_delta_ship<<<H,512>>>(Si,Sa,conv,g,beta,oa);
    k_delta_reg <<<H,512>>>(Si,Sb,conv,g,beta,ob);
    CK(cudaDeviceSynchronize());
    std::vector<float> a(STATE),b(STATE),ha(H*SK),hb2(H*SK);
    CK(cudaMemcpy(a.data(),Sa,STATE*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(),Sb,STATE*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ha.data(),oa,H*SK*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb2.data(),ob,H*SK*4,cudaMemcpyDeviceToHost));
    int sdiff=0,odiff=0; for(size_t i=0;i<STATE;i++) if(a[i]!=b[i])sdiff++;
    for(int i=0;i<H*SK;i++) if(ha[i]!=hb2[i])odiff++;
    printf("bitwise reg-vs-ship: state %d diffs, o %d diffs  %s\n", sdiff, odiff,
           (sdiff==0&&odiff==0)?"BITWISE IDENTICAL":"*** DIFFER ***");
    return 0;
}
