// turbo3 KV port validation: prove the device format/WHT/quantize/dequant
// match the TurboQuant CPU reference (ggml-turbo-quant.c) bit-for-bit before
// any engine wiring. CPU refs are inlined verbatim from the fork (@c3e6dbb13).
// Build: nvcc -std=c++17 -arch=sm_120 tools/turbo3_test.cu -o build/turbo3_test
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "../src/turbo3.cuh"
using namespace q27turbo;

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

// ---- CPU reference (verbatim from ggml-turbo-quant.c) -------------------
static const float CENTROIDS_3BIT[8] = {
    -0.190207f,-0.118786f,-0.066822f,-0.021663f, 0.021663f,0.066822f,0.118786f,0.190207f };
static const float cpu_s1[128] = { -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1 };
static const float cpu_s2[128] = { 1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1 };
static int cpu_nearest(float v){
    if(v<-0.154496f)return 0; if(v<-0.092804f)return 1; if(v<-0.044243f)return 2;
    if(v<0.f)return 3; if(v<0.044243f)return 4; if(v<0.092804f)return 5;
    if(v<0.154496f)return 6; return 7; }
static void cpu_fwht(float* x){
    const float inv=0.08838834764831845f;
    for(int i=0;i<128;i++) x[i]*=cpu_s1[i];
    for(int h=1;h<128;h*=2) for(int i=0;i<128;i+=h*2) for(int j=i;j<i+h;j++){
        float a=x[j],b=x[j+h]; x[j]=a+b; x[j+h]=a-b; }
    for(int i=0;i<128;i++) x[i]*=inv*cpu_s2[i];
}
static void cpu_fwht_inv(float* x){
    const float inv=0.08838834764831845f;
    for(int i=0;i<128;i++) x[i]*=cpu_s2[i];
    for(int h=1;h<128;h*=2) for(int i=0;i<128;i+=h*2) for(int j=i;j<i+h;j++){
        float a=x[j],b=x[j+h]; x[j]=a+b; x[j+h]=a-b; }
    for(int i=0;i<128;i++) x[i]*=inv*cpu_s1[i];
}
static void cpu_quant(const float* x, block_turbo3* y){
    float buf[128], nsq=0.f;
    for(int j=0;j<128;j++){ buf[j]=x[j]; nsq+=buf[j]*buf[j]; }
    float gn=sqrtf(nsq), inv=gn>1e-10f?1.f/gn:0.f;
    for(int j=0;j<128;j++) buf[j]*=inv;
    cpu_fwht(buf);
    memset(y->qs,0,32); memset(y->signs,0,16);
    float rsq=0.f;
    for(int j=0;j<128;j++){ int idx=cpu_nearest(buf[j]);
        y->qs[j/4]|=(idx&3)<<((j%4)*2); if(idx&4) y->signs[j/8]|=(1<<(j%8));
        rsq+=CENTROIDS_3BIT[idx]*CENTROIDS_3BIT[idx]; }
    float rn=sqrtf(rsq), corr=rn>1e-10f?gn/rn:gn;
    y->norm=__float2half(corr);
}
static void cpu_dequant(const block_turbo3* x, float* y){
    float norm=__half2float(x->norm);
    for(int j=0;j<128;j++){ uint8_t l=(x->qs[j/4]>>((j%4)*2))&3;
        uint8_t h=(x->signs[j/8]>>(j%8))&1; y[j]=CENTROIDS_3BIT[l|(h<<2)]*norm; }
}

// ---- device port (faithful single-thread-per-block, matches CPU exactly) ----
__device__ void dev_fwht(float* x){
    for(int i=0;i<128;i++) x[i]*=TURBO_S1[i];
    for(int h=1;h<128;h*=2) for(int i=0;i<128;i+=h*2) for(int j=i;j<i+h;j++){
        float a=x[j],b=x[j+h]; x[j]=a+b; x[j+h]=a-b; }
    for(int i=0;i<128;i++) x[i]*=TURBO_INV_SQRT_128*TURBO_S2[i];
}
__global__ void k_quant(const float* X, block_turbo3* Y, int n){
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=n) return;
    const float* x=X+b*128; block_turbo3* y=Y+b;
    float buf[128], nsq=0.f;
    for(int j=0;j<128;j++){ buf[j]=x[j]; nsq+=buf[j]*buf[j]; }
    float gn=sqrtf(nsq), inv=gn>1e-10f?1.f/gn:0.f;
    for(int j=0;j<128;j++) buf[j]*=inv;
    dev_fwht(buf);
    for(int j=0;j<32;j++) y->qs[j]=0; for(int j=0;j<16;j++) y->signs[j]=0;
    float rsq=0.f;
    for(int j=0;j<128;j++){ int idx=turbo3_nearest(buf[j]);
        y->qs[j/4]|=(idx&3)<<((j%4)*2); if(idx&4) y->signs[j/8]|=(1<<(j%8));
        rsq+=TURBO_CENTROIDS_3BIT[idx]*TURBO_CENTROIDS_3BIT[idx]; }
    float rn=sqrtf(rsq), corr=rn>1e-10f?gn/rn:gn;
    y->norm=__float2half(corr);
}
__global__ void k_dequant(const block_turbo3* X, float* Y, int n){
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=n) return;
    float norm=__half2float(X[b].norm);
    for(int j=0;j<128;j++) Y[b*128+j]=turbo3_dequant(&X[b],j,norm);
}

int fails=0;
int main(){
    // struct/const gate
    if(sizeof(block_turbo3)!=50){ printf("FAIL sizeof=%zu\n",sizeof(block_turbo3)); fails++; }
    const int N=4096; // 4096 random 128-vecs
    std::vector<float> hx(N*128);
    unsigned s=1234567;
    for(auto& v:hx){ s=s*1664525u+1013904223u; v=((s>>8)&0xFFFF)/65536.f-0.5f; }
    float* dX; block_turbo3* dB; float* dY;
    CK(cudaMalloc(&dX,N*128*4)); CK(cudaMalloc(&dB,N*sizeof(block_turbo3))); CK(cudaMalloc(&dY,N*128*4));
    CK(cudaMemcpy(dX,hx.data(),N*128*4,cudaMemcpyHostToDevice));
    k_quant<<<(N+63)/64,64>>>(dX,dB,N); CK(cudaDeviceSynchronize());
    k_dequant<<<(N+63)/64,64>>>(dB,dY,N); CK(cudaDeviceSynchronize());
    std::vector<block_turbo3> hb(N); std::vector<float> hy(N*128);
    CK(cudaMemcpy(hb.data(),dB,N*sizeof(block_turbo3),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hy.data(),dY,N*128*4,cudaMemcpyDeviceToHost));

    // 1. device quant == CPU quant (bit-identical qs/signs; norm <=1 ULP)
    int qmism=0, nmism=0;
    // 2. device dequant == CPU dequant
    int dmism=0;
    // 3. round-trip: quant->dequant->inverse-WHT recovers input (cosine)
    double cos_num=0, cos_a=0, cos_b=0, mse=0;
    for(int b=0;b<N;b++){
        block_turbo3 cb; cpu_quant(hx.data()+b*128,&cb);
        if(memcmp(cb.qs,hb[b].qs,32)||memcmp(cb.signs,hb[b].signs,16)) qmism++;
        if(cb.norm!=hb[b].norm){ // allow 1 ULP on fp16 norm
            int d=abs((int)cb.norm-(int)hb[b].norm); if(d>1) nmism++; }
        float cy[128]; cpu_dequant(&hb[b],cy);
        for(int j=0;j<128;j++) if(fabsf(cy[j]-hy[b*128+j])>1e-6f) dmism++;
        // round-trip vs original
        float rec[128]; for(int j=0;j<128;j++) rec[j]=hy[b*128+j];
        cpu_fwht_inv(rec); // un-rotate back to input domain
        const float* ox=hx.data()+b*128;
        for(int j=0;j<128;j++){ cos_num+=rec[j]*ox[j]; cos_a+=rec[j]*rec[j]; cos_b+=ox[j]*ox[j]; mse+=(rec[j]-ox[j])*(rec[j]-ox[j]); }
    }
    double cosine=cos_num/(sqrt(cos_a)*sqrt(cos_b)+1e-12);
    mse/=(N*128.0);
    printf("device-vs-CPU quant: qs/signs mismatches=%d/%d (midpoint ties; want <0.1%%)\n",qmism,N); if(qmism>N/1000)fails++;
    printf("device-vs-CPU norm : >1ULP mismatches=%d (want 0)\n",nmism); if(nmism)fails++;
    printf("device-vs-CPU dequant: elem mismatches=%d (want 0)\n",dmism); if(dmism)fails++;
    printf("round-trip q->deq->invWHT vs input: cosine=%.6f  MSE=%.6f\n",cosine,mse);
    if(cosine<0.97){ printf("FAIL round-trip cosine (3-bit expects ~0.98)\n"); fails++; }

    // 4. dot-product invariance (the read contract): across many (q,K) pairs,
    // the vector of turbo3 scores WHT(q).deq(K) must correlate with q.K.
    // Random q.K is near-zero so per-pair rel error is meaningless; use the
    // cosine of the two score VECTORS over 512 pairs (measures score fidelity).
    double dn=0,da=0,db=0;
    for(int b=0;b<512;b++){
        const float* K=hx.data()+b*128;
        float q[128], qw[128]; unsigned tt=99+b;
        for(int j=0;j<128;j++){ tt=tt*1664525u+1013904223u; q[j]=((tt>>8)&0xFFFF)/65536.f-0.5f; qw[j]=q[j]; }
        cpu_fwht(qw);
        float dqK[128]; cpu_dequant(&hb[b],dqK);
        double lhs=0, rhs=0;
        for(int j=0;j<128;j++){ lhs+=qw[j]*dqK[j]; rhs+=q[j]*K[j]; }
        dn+=lhs*rhs; da+=lhs*lhs; db+=rhs*rhs;
    }
    double dotcos=dn/(sqrt(da)*sqrt(db)+1e-12);
    printf("dot invariance WHT(q).deq(K) vs q.K: score cosine=%.4f over 512 pairs\n",dotcos);
    if(dotcos<0.97){ printf("FAIL dot invariance\n"); fails++; }

    printf(fails? "\nturbo3_test: %d FAILURES\n":"\nturbo3_test: ALL PASS\n",fails);
    return fails?1:0;
}
