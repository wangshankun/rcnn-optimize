#include <x86intrin.h>
#include <smmintrin.h>  /* SSE 4.1 */
#include <immintrin.h>  /* SSE 4.1 */

//格式与numpy一致i是列的index; j是行的index
#define A(j, i) a[ (j)*lda + (i) ]
#define B(j, i) b[ (j)*ldb + (i) ]
#define C(j, i) c[ (j)*ldc + (i) ]

void AddDot4x4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
    //printf("AddDot4x4_k4vs A:\r\n");
    //print_matrix(4,k,a,lda);
    //printf("AddDot4x4_k4vs B:\r\n");    
    //print_matrix(k,4,b,ldb);
    
    int p;
    v2df_t  c_00_vreg, c_10_vreg, c_20_vreg, c_30_vreg;
    c_00_vreg.v = _mm_load_ps((float *)&C(0, 0));
    c_10_vreg.v = _mm_load_ps((float *)&C(1, 0));
    c_20_vreg.v = _mm_load_ps((float *)&C(2, 0));
    c_30_vreg.v = _mm_load_ps((float *)&C(3, 0));
    
    for ( p = 0; p < k; p+=4 )
    {
        v2df_t a_00p_vreg, a_10p_vreg, a_20p_vreg, a_30p_vreg;
        a_00p_vreg.v = _mm_load_ps((float *)&A(0, p));
        a_10p_vreg.v = _mm_load_ps((float *)&A(1, p));
        a_20p_vreg.v = _mm_load_ps((float *)&A(2, p));
        a_30p_vreg.v = _mm_load_ps((float *)&A(3, p));
        
        v2df_t b_p00_vreg, b_p10_vreg, b_p20_vreg, b_p30_vreg;
        b_p00_vreg.v = _mm_load_ps((float *)&B(p + 0, 0));
        b_p10_vreg.v = _mm_load_ps((float *)&B(p + 1, 0));
        b_p20_vreg.v = _mm_load_ps((float *)&B(p + 2, 0));
        b_p30_vreg.v = _mm_load_ps((float *)&B(p + 3, 0));
        
        c_00_vreg.v = _mm_add_ps(c_00_vreg.v, _mm_mul_ps(_mm_set1_ps(a_00p_vreg.s[0]), b_p00_vreg.v));
        c_10_vreg.v = _mm_add_ps(c_10_vreg.v, _mm_mul_ps(_mm_set1_ps(a_10p_vreg.s[0]), b_p00_vreg.v));
        c_20_vreg.v = _mm_add_ps(c_20_vreg.v, _mm_mul_ps(_mm_set1_ps(a_20p_vreg.s[0]), b_p00_vreg.v));
        c_30_vreg.v = _mm_add_ps(c_30_vreg.v, _mm_mul_ps(_mm_set1_ps(a_30p_vreg.s[0]), b_p00_vreg.v));
        
        c_00_vreg.v = _mm_add_ps(c_00_vreg.v, _mm_mul_ps(_mm_set1_ps(a_00p_vreg.s[1]), b_p10_vreg.v));
        c_10_vreg.v = _mm_add_ps(c_10_vreg.v, _mm_mul_ps(_mm_set1_ps(a_10p_vreg.s[1]), b_p10_vreg.v));
        c_20_vreg.v = _mm_add_ps(c_20_vreg.v, _mm_mul_ps(_mm_set1_ps(a_20p_vreg.s[1]), b_p10_vreg.v));
        c_30_vreg.v = _mm_add_ps(c_30_vreg.v, _mm_mul_ps(_mm_set1_ps(a_30p_vreg.s[1]), b_p10_vreg.v));
        
        c_00_vreg.v = _mm_add_ps(c_00_vreg.v, _mm_mul_ps(_mm_set1_ps(a_00p_vreg.s[2]), b_p20_vreg.v));
        c_10_vreg.v = _mm_add_ps(c_10_vreg.v, _mm_mul_ps(_mm_set1_ps(a_10p_vreg.s[2]), b_p20_vreg.v));
        c_20_vreg.v = _mm_add_ps(c_20_vreg.v, _mm_mul_ps(_mm_set1_ps(a_20p_vreg.s[2]), b_p20_vreg.v));
        c_30_vreg.v = _mm_add_ps(c_30_vreg.v, _mm_mul_ps(_mm_set1_ps(a_30p_vreg.s[2]), b_p20_vreg.v));
        
        c_00_vreg.v = _mm_add_ps(c_00_vreg.v, _mm_mul_ps(_mm_set1_ps(a_00p_vreg.s[3]), b_p30_vreg.v));
        c_10_vreg.v = _mm_add_ps(c_10_vreg.v, _mm_mul_ps(_mm_set1_ps(a_10p_vreg.s[3]), b_p30_vreg.v));
        c_20_vreg.v = _mm_add_ps(c_20_vreg.v, _mm_mul_ps(_mm_set1_ps(a_20p_vreg.s[3]), b_p30_vreg.v));    
        c_30_vreg.v = _mm_add_ps(c_30_vreg.v, _mm_mul_ps(_mm_set1_ps(a_30p_vreg.s[3]), b_p30_vreg.v));  
    }
    _mm_store_ps(&(C( 0, 0 )), c_00_vreg.v);
    _mm_store_ps(&(C( 1, 0 )), c_10_vreg.v);
    _mm_store_ps(&(C( 2, 0 )), c_20_vreg.v);
    _mm_store_ps(&(C( 3, 0 )), c_30_vreg.v);
}

//A连续取4行, 4*k结构
void PackMatrixA(int k, float *a, int lda, float *a_to)
{
    int i;
    float 
    *a_0i_pntr = &A( 0, 0 ), *a_1i_pntr = &A( 1, 0 ),
    *a_2i_pntr = &A( 2, 0 ), *a_3i_pntr = &A( 3, 0 ),
    *a_0_to    = a_to + 0*k,
    *a_1_to    = a_to + 1*k,
    *a_2_to    = a_to + 2*k,
    *a_3_to    = a_to + 3*k;
   

    
    for( i=0; i<k; i++)
    {
        *a_0_to++ = *a_0i_pntr++;
        *a_1_to++ = *a_1i_pntr++;
        *a_2_to++ = *a_2i_pntr++;
        *a_3_to++ = *a_3i_pntr++;
    }
}
//B连续取4列, k*4结构
void PackMatrixB(int k, float *b, int ldb, float *b_to)
{
    int j;

    for(j = 0; j < k; j++)
    {
        float *b_ji_pntr = &B(j, 0);

        *(b_to + 0) = *(b_ji_pntr + 0);
        *(b_to + 1) = *(b_ji_pntr + 1);
        *(b_to + 2) = *(b_ji_pntr + 2);
        *(b_to + 3) = *(b_ji_pntr + 3);

        b_to += 4;
    }
}

void InnerKernelPack(int m, int n, int k, float *a, int lda, 
                                           float *b, int ldb,
                                           float *c, int ldc, int first_time)
{
    int i, j;
    float packedA[m * k], packedB[k * n];

    //printf("InnerKernelPack A:\r\n");
    //print_matrix(m,k,a,lda);
    //printf("InnerKernelPack B:\r\n");
    //print_matrix(k,n,b,ldb);
    
    for (j = 0; j < m; j+=4)
    {
        if(first_time) PackMatrixA(k, &A(j, 0), lda, &packedA[j * k]);
        for (i = 0; i < n; i+=4)
        {
            if(j == 0) PackMatrixB(k, &B(0, i), ldb, &packedB[k * i]);
            AddDot4x4(k, &packedA[j * k], k, &packedB[k * i], 4, &C(j, i), ldc);
        }
    }
}

#define PFIRST 512
#define PLAST  512
#define PINC   512

#define mc 64
#define kc 64
#define min( i, j ) ( (i)<(j) ? (i): (j) )

void MY_MMult_Inner_Pack(int m, int n, int k,  float *a, int lda, 
                                                float *b, int ldb,
                                                float *c, int ldc)
{
    int j, p, pb, jb;

    for (p = 0; p < k; p += kc)
    {
        pb = min(k - p, kc);
        for (j = 0; j< m; j += mc)
        {
            jb = min(m - j, mc);
            InnerKernelPack(jb, n, pb, &A(j, p), lda, &B(p, 0), ldb, &C(j, 0), ldc, j == 0);
        }
    }
}