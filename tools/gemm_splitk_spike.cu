// gemm_splitk_spike -- does split-K fill idle SMs on SHORT-prompt prefill?
//
// The prefill weight GEMM grid is (ceil(T/NT), ceil(rows/MR), 1). At deep
// prefill (T=65536) that grid is tens of thousands of blocks -- the SMs are
// saturated and split-K buys nothing. But the SUFFIX-round prefill on a warm
// agentic turn (prefix-cache miss on tens-to-hundreds of new tokens) runs at
// tiny T: grid.x collapses to 1, and for a small-output-rows weight
// (ffn_down: rows=5120 -> 80 row-tiles < 170 SMs) HALF the machine sits idle.
// Split-K on the K/stage axis (blockIdx.z) adds nsp blocks per output tile to
// fill it, at the cost of an nsp-wide float partial buffer + an ordered reduce.
//
// DETERMINISM: acc is float and the per-64/128-K group scale is applied before
// the cross-group add, so the K reduction is a FLOAT sum in a fixed stage
// order. Splitting K regroups that sum; float add is non-associative, so the
// result is NOT bitwise-identical to the single-CTA path -- this is a
// tolerance-gated speed lever (turbo3/fp8 class), NOT a canonical-gate path.
// The reduce is done in fixed sp-order so the delta is at least reproducible
// run-to-run. This spike measures both: the crossover T where split-K wins,
// and the numerical delta it costs.
//
// nsp==1 takes the direct-to-y path and is byte-identical to the shipped
// XG64/Q4 kernel (scalar-A + ldmatrix-B), so it is the honest baseline.
//
// Build (5090 only, fast):
//   nvcc -O2 -std=c++17 -gencode arch=compute_120,code=sm_120 \
//     tools/gemm_splitk_spike.cu src/device_model.cu src/loader.cpp -o build/gemm_splitk_spike
// Usage: gemm_splitk_spike model.q27 [weight_name]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../src/device_model.h"
#include "../src/loader.h"
#include "../src/kernels.cuh"
#define CK(x) do{cudaError_t e=(x);if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

static __device__ __forceinline__ void mma_s8(int&d0,int&d1,int&d2,int&d3,uint32_t a0,uint32_t a1,
                                              uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    const int z=0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(z),"r"(z),"r"(z),"r"(z));
}
static __device__ __forceinline__ void mma_s8_acc(int&d0,int&d1,int&d2,int&d3,uint32_t a0,uint32_t a1,
                                                  uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
static __device__ __forceinline__ void ldm_x2(uint32_t&r0,uint32_t&r1,const void*p){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];\n"
                 :"=r"(r0),"=r"(r1):"r"(a));
}

// SHIPPED XG64/Q4 path (scalar-A + ldmatrix-B), with a K-split on blockIdx.z.
// Each z-block owns a contiguous stage range [st_lo, st_hi). When nsp==1 it
// writes acc straight to y (== production). When nsp>1 it writes its float
// partial to tmp[sp]; k_sk_reduce sums the nsp partials in sp order into y.
template<int NT>
__global__ __launch_bounds__(256) void k_gemm_sk(const uint8_t* __restrict__ W,const __half* __restrict__ S,
        const int8_t* __restrict__ nat,const float* __restrict__ xs,
        float* __restrict__ y,float* __restrict__ tmp,int64_t rows,int64_t cols,int T){
    constexpr int MR=64,KS=128,XGS=64,XSC=KS/XGS,TS=NT/16,LDW=KS+16,LDX=KS+16;
    extern __shared__ unsigned char smem_raw[];
    int8_t* s_w=(int8_t*)smem_raw; int8_t* s_x=(int8_t*)(s_w+MR*LDW);
    float* s_ws=(float*)(s_x+NT*LDX); float* s_xs=(float*)(s_ws+MR*2);
    const int warp=threadIdx.x/32,lane=threadIdx.x&31,wm=warp%4,wn=warp/4,gid=lane>>2,tg=lane&3;
    const int64_t r0=(int64_t)blockIdx.y*MR; const int t0=blockIdx.x*NT; const int n_stages=(int)(cols/KS);
    const int nsp=gridDim.z, sp=blockIdx.z;
    const int chunk=(n_stages+nsp-1)/nsp;
    const int st_lo=sp*chunk; int st_hi=st_lo+chunk; if(st_hi>n_stages)st_hi=n_stages;
    float acc[TS][4];
    #pragma unroll
    for(int s=0;s<TS;s++)for(int e=0;e<4;e++)acc[s][e]=0.f;
    constexpr int WLD=MR*(KS/2)/4/256, XLD=NT*KS/4/256, XSL=(NT*XSC+255)/256;
    const int tid=threadIdx.x,nws=MR*2;
    uint32_t rw[WLD],rx[XLD]; float rws=0.f,rxs[XSL>0?XSL:1];
    auto load_stage=[&](int st){
        const int64_t k0=(int64_t)st*KS;
        #pragma unroll
        for(int i=0;i<WLD;i++){int idx=i*256+tid,rr=idx/16,pb4=idx%16;
            rw[i]=r0+rr<rows?__ldg((const uint32_t*)(W+(r0+rr)*(cols/2)+k0/2)+pb4):0x88888888u;}
        #pragma unroll
        for(int i=0;i<XLD;i++){int idx=i*256+tid,tt=idx/(KS/4),u=idx%(KS/4);
            rx[i]=t0+tt<T?__ldg((const uint32_t*)(nat+(size_t)(t0+tt)*cols+k0)+u):0u;}
        if(tid<nws){int rr=tid/2,g=tid%2; rws=r0+rr<rows?__half2float(__ldg(S+(r0+rr)*(cols/64)+k0/64+g)):0.f;}
        #pragma unroll
        for(int i=0;i<XSL;i++){int idx=i*256+tid,tt=idx/XSC,cc=idx%XSC;
            rxs[i]=(idx<NT*XSC&&t0+tt<T)?__ldg(xs+(size_t)(t0+tt)*(cols/XGS)+k0/XGS+cc):0.f;}
    };
    auto store_stage=[&](){
        #pragma unroll
        for(int i=0;i<WLD;i++){int idx=i*256+tid,rr=idx/16,pb4=idx%16; int8_t* dst=s_w+rr*LDW+pb4*8;
            const uint32_t p=rw[i],lo=p&0x0F0F0F0Fu,hi=(p>>4)&0x0F0F0F0Fu;
            *(uint32_t*)dst=__vsub4(__byte_perm(lo,hi,0x5140),0x08080808u);
            *(uint32_t*)(dst+4)=__vsub4(__byte_perm(lo,hi,0x7362),0x08080808u);}
        #pragma unroll
        for(int i=0;i<XLD;i++){int idx=i*256+tid,tt=idx/(KS/4),u=idx%(KS/4);
            *(uint32_t*)(s_x+tt*LDX+u*4)=rx[i];}
        if(tid<nws)s_ws[tid]=rws;
        #pragma unroll
        for(int i=0;i<XSL;i++){int idx=i*256+tid; if(idx<NT*XSC)s_xs[idx]=rxs[i];}
    };
    // Empty range (nsp > n_stages): acc stays 0, skip the loads (k0 would be OOB).
    if(st_lo<st_hi){
        load_stage(st_lo);
        for(int st=st_lo;st<st_hi;st++){
            __syncthreads(); store_stage(); if(st+1<st_hi)load_stage(st+1); __syncthreads();
            #pragma unroll
            for(int gg=0;gg<2;gg++){
                const int kb=gg*64;
                const int8_t* w0=s_w+(wm*16+gid)*LDW+kb;
                uint32_t a0=*(const uint32_t*)(w0+tg*4),      a1=*(const uint32_t*)(w0+8*LDW+tg*4);
                uint32_t a2=*(const uint32_t*)(w0+tg*4+16),   a3=*(const uint32_t*)(w0+8*LDW+tg*4+16);
                uint32_t a4=*(const uint32_t*)(w0+tg*4+32),   a5=*(const uint32_t*)(w0+8*LDW+tg*4+32);
                uint32_t a6=*(const uint32_t*)(w0+tg*4+48),   a7=*(const uint32_t*)(w0+8*LDW+tg*4+48);
                const float wsc0=s_ws[(wm*16+gid)*2+gg], wsc1=s_ws[(wm*16+gid+8)*2+gg];
                #pragma unroll
                for(int s=0;s<TS;s++){
                    const int tb=wn*(NT/2)+s*8;
                    uint32_t b0,b1,b2,b3;
                    const int8_t* xr=s_x+(tb+(lane%8))*LDX+kb+((lane%16)/8)*16;
                    ldm_x2(b0,b1, xr);
                    ldm_x2(b2,b3, xr+32);
                    int d0,d1,d2,d3;
                    mma_s8(d0,d1,d2,d3,a0,a1,a2,a3,b0,b1);
                    mma_s8_acc(d0,d1,d2,d3,a4,a5,a6,a7,b2,b3);
                    const float xs0=s_xs[(tb+tg*2)*2+gg], xs1=s_xs[(tb+tg*2+1)*2+gg];
                    acc[s][0]+=wsc0*xs0*(float)d0; acc[s][1]+=wsc0*xs1*(float)d1;
                    acc[s][2]+=wsc1*xs0*(float)d2; acc[s][3]+=wsc1*xs1*(float)d3;
                }
            }
        }
    }
    float* out = (nsp==1) ? y : (tmp + (size_t)sp*(size_t)T*rows);
    const int64_t row0=r0+wm*16+gid;
    #pragma unroll
    for(int s=0;s<TS;s++){const int tok0=t0+wn*(NT/2)+s*8+tg*2;
        #pragma unroll
        for(int e=0;e<4;e++){int64_t row=row0+(e>=2?8:0);int tok=tok0+(e&1);
            if(row<rows&&tok<T)out[(size_t)tok*rows+row]=acc[s][e];}}
}

// Ordered reduce: y[i] = sum_{sp=0..nsp-1} tmp[sp][i], fixed sp order.
__global__ void k_sk_reduce(const float* __restrict__ tmp,float* __restrict__ y,
                            size_t n,int nsp,int64_t stride){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){
        float a=0.f;
        for(int sp=0;sp<nsp;sp++) a+=tmp[(size_t)sp*stride+i];
        y[i]=a;
    }
}

template<int NT>
static double run_nt(const q27::DevTensor& w,const int8_t* nat,const float* s64,float* y,float* tmp,
                     int T,int nsp,int reps){
    constexpr int MR=64,KS=128,LDW=KS+16,LDX=KS+16,XSC=2;
    size_t SM=(size_t)MR*LDW+(size_t)NT*LDX+(MR*2+NT*XSC)*4;
    static bool a=false;
    if(!a){CK(cudaFuncSetAttribute(k_gemm_sk<NT>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SM));a=true;}
    dim3 g((unsigned)((T+NT-1)/NT),(unsigned)((w.rows+MR-1)/MR),(unsigned)nsp);
    size_t nout=(size_t)T*w.rows;
    auto once=[&](){
        k_gemm_sk<NT><<<g,256,SM>>>((const uint8_t*)w.data,(const __half*)w.scales,nat,s64,y,tmp,w.rows,w.cols,T);
        if(nsp>1){int th=256,bl=(int)((nout+th-1)/th); if(bl>65535)bl=65535;
            k_sk_reduce<<<bl,th>>>(tmp,y,nout,nsp,(int64_t)T*w.rows);}
    };
    once(); CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0)); for(int r=0;r<reps;r++)once(); CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); return ms/reps;
}

// dispatch NT like production: smallest tile that covers T
static double run(const q27::DevTensor& w,const int8_t* nat,const float* s64,float* y,float* tmp,
                  int T,int nsp,int reps){
    int nt = T<=16?16 : T<=32?32 : T<=64?64 : 128;
    switch(nt){
        case 16: return run_nt<16>(w,nat,s64,y,tmp,T,nsp,reps);
        case 32: return run_nt<32>(w,nat,s64,y,tmp,T,nsp,reps);
        case 64: return run_nt<64>(w,nat,s64,y,tmp,T,nsp,reps);
        default: return run_nt<128>(w,nat,s64,y,tmp,T,nsp,reps);
    }
}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s model.q27 [weight_name]\n",argv[0]);return 1;}
    const char* wname = argc>=3 ? argv[2] : "blk.0.ffn_down.weight";
    q27::Model m=q27::Model::open(argv[1]); q27::DeviceModel dm(m);
    const q27::DevTensor& w=dm.upload(wname);
    int64_t rows=w.rows,cols=w.cols;
    int nsm=0; { cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0)); nsm=p.multiProcessorCount; }
    printf("weight %s  %ldx%ld  (rows/MR=%ld row-tiles, cols/KS=%ld stages)  SMs=%d\n\n",
           wname,(long)rows,(long)cols,(rows+63)/64,cols/128,nsm);

    const int Ts[]={16,32,64,128,256,1024};
    const int NSPs[]={1,2,4,8};
    const int maxT=1024, maxNsp=8;
    // g64 int8 activations sized for maxT
    int8_t* nat; float* s64;
    CK(cudaMalloc(&nat,(size_t)maxT*cols)); CK(cudaMalloc(&s64,(size_t)maxT*(cols/64)*4));
    { std::vector<int8_t> hn((size_t)maxT*cols); std::vector<float> hs((size_t)maxT*(cols/64));
      for(size_t i=0;i<hn.size();i++) hn[i]=(int8_t)(((i*2654435761u)>>21)%127-63);
      for(size_t i=0;i<hs.size();i++) hs[i]=0.01f+0.001f*(i%17);
      CK(cudaMemcpy(nat,hn.data(),hn.size(),cudaMemcpyHostToDevice));
      CK(cudaMemcpy(s64,hs.data(),hs.size()*4,cudaMemcpyHostToDevice)); }
    float *y; CK(cudaMalloc(&y,(size_t)maxT*rows*4));
    float *tmp; CK(cudaMalloc(&tmp,(size_t)maxNsp*maxT*rows*4));
    std::vector<float> ref((size_t)maxT*rows), got((size_t)maxT*rows);

    printf("%6s %5s %5s %10s %8s %9s %s\n","T","NT","nsp","ms","speedup","rel-err","diff");
    for(int T : Ts){
        int nt = T<=16?16 : T<=32?32 : T<=64?64 : 128;
        double t1=0;
        for(int nsp : NSPs){
            double ms=run(w,nat,s64,y,tmp,T,nsp,100);
            CK(cudaMemcpy(got.data(),y,(size_t)T*rows*4,cudaMemcpyDeviceToHost));
            if(nsp==1){ t1=ms; ref=got; }
            double num=0,den=0; long diff=0;
            for(size_t i=0;i<(size_t)T*rows;i++){double d=(double)got[i]-ref[i];num+=d*d;den+=(double)ref[i]*ref[i];
                if(got[i]!=ref[i])diff++;}
            double rel=den>0?sqrt(num/den):0;
            printf("%6d %5d %5d %10.4f %7.2fx %9.2e %ld/%ld %s\n",
                   T,nt,nsp,ms,t1/ms,rel,diff,(long)((size_t)T*rows),
                   nsp==1?"(baseline)":(diff==0?"BITWISE":"tol"));
        }
        printf("\n");
    }
    return 0;
}
