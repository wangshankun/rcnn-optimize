#include <x86intrin.h>
#include <smmintrin.h>  /* SSE 4.1 */
#include <immintrin.h>  /* SSE 4.1 */

//格式与numpy一致i是列的index; j是行的index
#define A(j, i) a[ (j)*lda + (i) ]
#define B(j, i) b[ (j)*ldb + (i) ]
#define C(j, i) c[ (j)*ldc + (i) ]

void AddDot4x4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
    int p;
    v2df_t  c_00_vreg, c_10_vreg, c_20_vreg, c_30_vreg;
    c_00_vreg.v = _mm_setzero_ps();
    c_10_vreg.v = _mm_setzero_ps();
    c_20_vreg.v = _mm_setzero_ps();
    c_30_vreg.v = _mm_setzero_ps();
    
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

void MY_MMult( int m, int n, int k, float *a, int lda, 
                                       float *b, int ldb,
                                       float *c, int ldc )
{
    int i, j;

    for (j = 0; j < m; j+=4)
    {
        for (i = 0; i < n; i+=4)
        {   //A每次出一行的行首，B每次出一列的列首
            AddDot4x4(k, &A(j, 0 ), lda, &B(0, i), ldb, &C(j, i), ldc );
        }
    }
}