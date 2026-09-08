// Vendor-ceiling probe: cuBLASLt dense GEMM throughput on this card for the
// prefill projection shapes, in int8 (int32 acc), fp8 e4m3 (fp32 acc, fp16 out)
// and fp16 (fp32 acc). Gives the achievable percent-of-peak reference the
// 2026-08-17 prefill plan's P0 asked for. Build:
//   nvcc -O2 -arch=sm_120a cublaslt_peak.cu -lcublasLt -lcublas -o cublaslt_peak
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto _e = (x); if (_e != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublas err %d at %d\n", (int)_e, __LINE__); exit(1);} } while (0)
#define CU(x) do { auto _e = (x); if (_e != cudaSuccess) { fprintf(stderr, "cuda err %s at %d\n", cudaGetErrorString(_e), __LINE__); exit(1);} } while (0)

struct Shape { const char* name; int N, K; };

// C[M,N] = op(A) * B, TN layout as the int8/fp8 IMMA paths require:
// A stored K x M col-major (== M x K row-major) with op T; B stored K x N col-major, op N.
static double run(cublasLtHandle_t h, cudaDataType ta, cudaDataType tc, cublasComputeType_t ct, cudaDataType st,
                  int M, int N, int K, void* dA, void* dB, void* dC, void* ws, size_t wsz, cudaStream_t s) {
  cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, ct, st));
  cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
  cublasLtMatrixLayout_t la, lb, lc;
  CK(cublasLtMatrixLayoutCreate(&la, ta, K, M, K));
  CK(cublasLtMatrixLayoutCreate(&lb, ta, K, N, K));
  CK(cublasLtMatrixLayoutCreate(&lc, tc, M, N, M));
  cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
  cublasLtMatmulHeuristicResult_t res[8]; int nres = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, lc, lc, pref, 8, res, &nres));
  if (nres == 0) { fprintf(stderr, "no algo for %dx%dx%d\n", M, N, K); return 0; }
  float alpha_f = 1.f, beta_f = 0.f; int alpha_i = 1, beta_i = 0;
  const void* alpha = (st == CUDA_R_32I) ? (const void*)&alpha_i : (const void*)&alpha_f;
  const void* beta  = (st == CUDA_R_32I) ? (const void*)&beta_i  : (const void*)&beta_f;
  double best = 0;
  for (int a = 0; a < nres && a < 4; a++) {
    for (int i = 0; i < 3; i++)
      CK(cublasLtMatmul(h, op, alpha, dA, la, dB, lb, beta, dC, lc, dC, lc, &res[a].algo, ws, wsz, s));
    cudaEvent_t e0, e1; CU(cudaEventCreate(&e0)); CU(cudaEventCreate(&e1));
    const int it = 20;
    CU(cudaEventRecord(e0, s));
    for (int i = 0; i < it; i++)
      CK(cublasLtMatmul(h, op, alpha, dA, la, dB, lb, beta, dC, lc, dC, lc, &res[a].algo, ws, wsz, s));
    CU(cudaEventRecord(e1, s)); CU(cudaEventSynchronize(e1));
    float ms; CU(cudaEventElapsedTime(&ms, e0, e1));
    double tflops = 2.0 * M * N * K * it / (ms * 1e-3) / 1e12;
    if (tflops > best) best = tflops;
    CU(cudaEventDestroy(e0)); CU(cudaEventDestroy(e1));
  }
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(op);
  return best;
}

int main(int argc, char** argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  CU(cudaSetDevice(dev));
  cudaDeviceProp p; CU(cudaGetDeviceProperties(&p, dev));
  int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev); printf("device %d: %s, %d SMs, %d MHz\n", dev, p.name, p.multiProcessorCount, clk / 1000);
  cublasLtHandle_t h; CK(cublasLtCreate(&h));
  cudaStream_t s; CU(cudaStreamCreate(&s));
  // Qwen3.8-27B projection shapes (N = out features, K = in features)
  Shape shapes[] = { {"attn_qkv+gate (N=16k,K=5k)", 16384, 5120}, {"attn_out (N=5k,K=8k)", 5120, 8192},
                     {"ffn_gate/up (N=17k,K=5k)", 17408, 5120}, {"ffn_down (N=5k,K=17k)", 5120, 17408} };
  int Ms[] = {1024, 4096};
  size_t maxA = 4096ull * 17408, maxB = 17408ull * 17408, maxC = 4096ull * 17408 * 4;
  void *dA, *dB, *dC, *ws; size_t wsz = 64ull << 20;
  CU(cudaMalloc(&dA, maxA * 2)); CU(cudaMalloc(&dB, maxB * 2)); CU(cudaMalloc(&dC, maxC)); CU(cudaMalloc(&ws, wsz));
  CU(cudaMemset(dA, 0x11, maxA * 2)); CU(cudaMemset(dB, 0x22, maxB * 2));
  printf("%-30s %6s %10s %10s %10s\n", "shape", "M", "int8 TOPS", "fp8 TFLOPS", "fp16 TFLOPS");
  for (auto& sh : shapes) for (int M : Ms) {
    double i8 = run(h, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I, M, sh.N, sh.K, dA, dB, dC, ws, wsz, s);
    double f8 = run(h, CUDA_R_8F_E4M3, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F, M, sh.N, sh.K, dA, dB, dC, ws, wsz, s);
    double f16 = run(h, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F, M, sh.N, sh.K, dA, dB, dC, ws, wsz, s);
    printf("%-30s %6d %10.0f %10.0f %10.0f\n", sh.name, M, i8, f8, f16);
  }
  return 0;
}
