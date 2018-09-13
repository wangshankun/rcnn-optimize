#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <smmintrin.h>  /* SSE 4.1 */
#include <immintrin.h>  /* SSE 4.1 */

//格式与numpy一致i是列的index; j是行的index
#define A(j, i) a[ (j)*lda + (i) ]
#define B(j, i) b[ (j)*ldb + (i) ]
#define C(j, i) c[ (j)*ldc + (i) ]

void print_matrix( int m, int n, float *a, int lda )
{
  int i, j;

  for ( j=0; j<m; j++ ){
    for ( i=0; i<n; i++ )
      printf("%04.0f ", A(j, i) );
    printf("\n");
  }
  printf("\n");
}

void random_matrix( int m, int n, float *a, int lda )
{
    int i,j,t=0;

    srand48(time(0));
    for (j = 0; j < m; j++ )
    {
        for ( i = 0; i < n; i++ )
        {
            //A(j, i) = 2.0 * (float)drand48( ) - 1.0;
            A(j, i) = t++;
        }
    }
}

void native_c( int m, int n, int k, float *a, int lda, 
                                     float *b, int ldb,
                                     float *c, int ldc )
{
    int i, j, p;

    for (j = 0; j < m; j++)
    {
        for (i = 0; i < n; i++)
        {        
            for (p = 0; p < k; p++)
            {      
                C(j, i) = C(j, i) + A(j, p) * B(p, i);
                //printf("C:%f A:%f B:%f\r\n",(C(j, i)),(A(j, p)),(B(p, i)));
            }
        }
    }
}


void AddDot4x4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
    int p;

    for ( p = 0; p < k; p++ )
    {
        /* First row */
        C( 0, 0 ) += A( 0, p ) * B( p, 0 );     
        C( 0, 1 ) += A( 0, p ) * B( p, 1 );     
        C( 0, 2 ) += A( 0, p ) * B( p, 2 );     
        C( 0, 3 ) += A( 0, p ) * B( p, 3 );     

        /* Second row */
        C( 1, 0 ) += A( 1, p ) * B( p, 0 );     
        C( 1, 1 ) += A( 1, p ) * B( p, 1 );     
        C( 1, 2 ) += A( 1, p ) * B( p, 2 );     
        C( 1, 3 ) += A( 1, p ) * B( p, 3 );     

        /* Third row */
        C( 2, 0 ) += A( 2, p ) * B( p, 0 );     
        C( 2, 1 ) += A( 2, p ) * B( p, 1 );     
        C( 2, 2 ) += A( 2, p ) * B( p, 2 );     
        C( 2, 3 ) += A( 2, p ) * B( p, 3 );     

        /* Fourth row */
        C( 3, 0 ) += A( 3, p ) * B( p, 0 );     
        C( 3, 1 ) += A( 3, p ) * B( p, 1 );     
        C( 3, 2 ) += A( 3, p ) * B( p, 2 );     
        C( 3, 3 ) += A( 3, p ) * B( p, 3 );     
    }
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


void AddDot4x4_k4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
    int p;
    for ( p = 0; p < k; p+=4 )
    {
        /* First row */
        C( 0, 0 ) += A( 0, p + 0) * B( p + 0, 0 );     
        C( 0, 1 ) += A( 0, p + 0) * B( p + 0, 1 );     
        C( 0, 2 ) += A( 0, p + 0) * B( p + 0, 2 );     
        C( 0, 3 ) += A( 0, p + 0) * B( p + 0, 3 );     
        C( 0, 0 ) += A( 0, p + 1) * B( p + 1, 0 );     
        C( 0, 1 ) += A( 0, p + 1) * B( p + 1, 1 );     
        C( 0, 2 ) += A( 0, p + 1) * B( p + 1, 2 );     
        C( 0, 3 ) += A( 0, p + 1) * B( p + 1, 3 ); 
        C( 0, 0 ) += A( 0, p + 2) * B( p + 2, 0 );     
        C( 0, 1 ) += A( 0, p + 2) * B( p + 2, 1 );     
        C( 0, 2 ) += A( 0, p + 2) * B( p + 2, 2 );     
        C( 0, 3 ) += A( 0, p + 2) * B( p + 2, 3 ); 
        C( 0, 0 ) += A( 0, p + 3) * B( p + 3, 0 );     
        C( 0, 1 ) += A( 0, p + 3) * B( p + 3, 1 );     
        C( 0, 2 ) += A( 0, p + 3) * B( p + 3, 2 );     
        C( 0, 3 ) += A( 0, p + 3) * B( p + 3, 3 ); 
        
        /* Second row */
        C( 1, 0 ) += A( 1, p + 0) * B( p + 0, 0 );     
        C( 1, 1 ) += A( 1, p + 0) * B( p + 0, 1 );     
        C( 1, 2 ) += A( 1, p + 0) * B( p + 0, 2 );     
        C( 1, 3 ) += A( 1, p + 0) * B( p + 0, 3 );     
        C( 1, 0 ) += A( 1, p + 1) * B( p + 1, 0 );     
        C( 1, 1 ) += A( 1, p + 1) * B( p + 1, 1 );     
        C( 1, 2 ) += A( 1, p + 1) * B( p + 1, 2 );     
        C( 1, 3 ) += A( 1, p + 1) * B( p + 1, 3 ); 
        C( 1, 0 ) += A( 1, p + 2) * B( p + 2, 0 );     
        C( 1, 1 ) += A( 1, p + 2) * B( p + 2, 1 );     
        C( 1, 2 ) += A( 1, p + 2) * B( p + 2, 2 );     
        C( 1, 3 ) += A( 1, p + 2) * B( p + 2, 3 ); 
        C( 1, 0 ) += A( 1, p + 3) * B( p + 3, 0 );     
        C( 1, 1 ) += A( 1, p + 3) * B( p + 3, 1 );     
        C( 1, 2 ) += A( 1, p + 3) * B( p + 3, 2 );     
        C( 1, 3 ) += A( 1, p + 3) * B( p + 3, 3 );     

        /* Third row */
        C( 2, 0 ) += A( 2, p + 0) * B( p + 0, 0 );     
        C( 2, 1 ) += A( 2, p + 0) * B( p + 0, 1 );     
        C( 2, 2 ) += A( 2, p + 0) * B( p + 0, 2 );     
        C( 2, 3 ) += A( 2, p + 0) * B( p + 0, 3 );     
        C( 2, 0 ) += A( 2, p + 1) * B( p + 1, 0 );     
        C( 2, 1 ) += A( 2, p + 1) * B( p + 1, 1 );     
        C( 2, 2 ) += A( 2, p + 1) * B( p + 1, 2 );     
        C( 2, 3 ) += A( 2, p + 1) * B( p + 1, 3 ); 
        C( 2, 0 ) += A( 2, p + 2) * B( p + 2, 0 );     
        C( 2, 1 ) += A( 2, p + 2) * B( p + 2, 1 );     
        C( 2, 2 ) += A( 2, p + 2) * B( p + 2, 2 );     
        C( 2, 3 ) += A( 2, p + 2) * B( p + 2, 3 ); 
        C( 2, 0 ) += A( 2, p + 3) * B( p + 3, 0 );     
        C( 2, 1 ) += A( 2, p + 3) * B( p + 3, 1 );     
        C( 2, 2 ) += A( 2, p + 3) * B( p + 3, 2 );     
        C( 2, 3 ) += A( 2, p + 3) * B( p + 3, 3 );      

        /* Fourth row */
        C( 3, 0 ) += A( 3, p + 0) * B( p + 0, 0 );     
        C( 3, 1 ) += A( 3, p + 0) * B( p + 0, 1 );     
        C( 3, 2 ) += A( 3, p + 0) * B( p + 0, 2 );     
        C( 3, 3 ) += A( 3, p + 0) * B( p + 0, 3 );     
        C( 3, 0 ) += A( 3, p + 1) * B( p + 1, 0 );     
        C( 3, 1 ) += A( 3, p + 1) * B( p + 1, 1 );     
        C( 3, 2 ) += A( 3, p + 1) * B( p + 1, 2 );     
        C( 3, 3 ) += A( 3, p + 1) * B( p + 1, 3 ); 
        C( 3, 0 ) += A( 3, p + 2) * B( p + 2, 0 );     
        C( 3, 1 ) += A( 3, p + 2) * B( p + 2, 1 );     
        C( 3, 2 ) += A( 3, p + 2) * B( p + 2, 2 );     
        C( 3, 3 ) += A( 3, p + 2) * B( p + 2, 3 ); 
        C( 3, 0 ) += A( 3, p + 3) * B( p + 3, 0 );     
        C( 3, 1 ) += A( 3, p + 3) * B( p + 3, 1 );     
        C( 3, 2 ) += A( 3, p + 3) * B( p + 3, 2 );     
        C( 3, 3 ) += A( 3, p + 3) * B( p + 3, 3 );    
    }
}

void MY_MMult_k4( int m, int n, int k, float *a, int lda, 
                                       float *b, int ldb,
                                       float *c, int ldc )
{
    int i, j;

    for (j = 0; j < m; j+=4)
    {
        for (i = 0; i < n; i+=4)
        {   //A每次出一行的行首，B每次出一列的列首
            AddDot4x4_k4(k, &A(j, 0 ), lda, &B(0, i), ldb, &C(j, i), ldc );
        }
    }
}

typedef union
{
  __m128  v;
  float   s[4];
} v2df_t;

void AddDot4x4_k4v( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
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
        c_00_vreg.v = _mm_add_ps(c_00_vreg.v, _mm_mul_ps(_mm_set1_ps(a_00p_vreg.s[1]), b_p10_vreg.v));
        c_00_vreg.v = _mm_add_ps(c_00_vreg.v, _mm_mul_ps(_mm_set1_ps(a_00p_vreg.s[2]), b_p20_vreg.v));
        c_00_vreg.v = _mm_add_ps(c_00_vreg.v, _mm_mul_ps(_mm_set1_ps(a_00p_vreg.s[3]), b_p30_vreg.v));
                   
        c_10_vreg.v = _mm_add_ps(c_10_vreg.v, _mm_mul_ps(_mm_set1_ps(a_10p_vreg.s[0]), b_p00_vreg.v));
        c_10_vreg.v = _mm_add_ps(c_10_vreg.v, _mm_mul_ps(_mm_set1_ps(a_10p_vreg.s[1]), b_p10_vreg.v));
        c_10_vreg.v = _mm_add_ps(c_10_vreg.v, _mm_mul_ps(_mm_set1_ps(a_10p_vreg.s[2]), b_p20_vreg.v));
        c_10_vreg.v = _mm_add_ps(c_10_vreg.v, _mm_mul_ps(_mm_set1_ps(a_10p_vreg.s[3]), b_p30_vreg.v));
  
        c_20_vreg.v = _mm_add_ps(c_20_vreg.v, _mm_mul_ps(_mm_set1_ps(a_20p_vreg.s[0]), b_p00_vreg.v));
        c_20_vreg.v = _mm_add_ps(c_20_vreg.v, _mm_mul_ps(_mm_set1_ps(a_20p_vreg.s[1]), b_p10_vreg.v));
        c_20_vreg.v = _mm_add_ps(c_20_vreg.v, _mm_mul_ps(_mm_set1_ps(a_20p_vreg.s[2]), b_p20_vreg.v));
        c_20_vreg.v = _mm_add_ps(c_20_vreg.v, _mm_mul_ps(_mm_set1_ps(a_20p_vreg.s[3]), b_p30_vreg.v));    

        c_30_vreg.v = _mm_add_ps(c_30_vreg.v, _mm_mul_ps(_mm_set1_ps(a_30p_vreg.s[0]), b_p00_vreg.v));
        c_30_vreg.v = _mm_add_ps(c_30_vreg.v, _mm_mul_ps(_mm_set1_ps(a_30p_vreg.s[1]), b_p10_vreg.v));
        c_30_vreg.v = _mm_add_ps(c_30_vreg.v, _mm_mul_ps(_mm_set1_ps(a_30p_vreg.s[2]), b_p20_vreg.v));
        c_30_vreg.v = _mm_add_ps(c_30_vreg.v, _mm_mul_ps(_mm_set1_ps(a_30p_vreg.s[3]), b_p30_vreg.v));  
    }
    _mm_store_ps(&(C( 0, 0 )), c_00_vreg.v);
    _mm_store_ps(&(C( 1, 0 )), c_10_vreg.v);
    _mm_store_ps(&(C( 2, 0 )), c_20_vreg.v);
    _mm_store_ps(&(C( 3, 0 )), c_30_vreg.v);
}

void MY_MMult_k4v( int m, int n, int k, float *a, int lda, 
                                       float *b, int ldb,
                                       float *c, int ldc )
{
    int i, j;

    for (j = 0; j < m; j+=4)
    {
        for (i = 0; i < n; i+=4)
        {   //A每次出一行的行首，B每次出一列的列首
            AddDot4x4_k4v(k, &A(j, 0 ), lda, &B(0, i), ldb, &C(j, i), ldc );
        }
    }
}


void AddDot4x4_k4vs( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
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

void InnerKernel(int m, int n, int k, float *a, int lda, 
                                       float *b, int ldb,
                                       float *c, int ldc)
{
    int i, j;

    for (j = 0; j < m; j+=4)
    {
        for (i = 0; i < n; i+=4)
        {   //A每次出一行的行首，B每次出一列的列首
            AddDot4x4_k4vs(k, &A(j, 0), lda, &B(0, i), ldb, &C(j, i), ldc);
        }
    }
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
                                           float *c, int ldc)
{
    int i, j;
    float packedA[m * k], packedB[k * n];

    //printf("InnerKernelPack A:\r\n");
    //print_matrix(m,k,a,lda);
    //printf("InnerKernelPack B:\r\n");
    //print_matrix(k,n,b,ldb);
    
    for (j = 0; j < m; j+=4)
    {
        PackMatrixA(k, &A(j, 0), lda, &packedA[j * k]);
        for (i = 0; i < n; i+=4)
        {
            if(j == 0) PackMatrixB(k, &B(0, i), ldb, &packedB[k * i]);
            AddDot4x4_k4vs(k, &packedA[j * k], k, &packedB[k * i], 4, &C(j, i), ldc);
        }
    }
}

void InnerKernelQPack(int m, int n, int k, float *a, int lda, 
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
            AddDot4x4_k4vs(k, &packedA[j * k], k, &packedB[k * i], 4, &C(j, i), ldc);
        }
    }
}

#define PFIRST 1024
#define PLAST  1024
#define PINC   1024

#define mc 128
#define kc 128
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
            InnerKernelPack(jb, n, pb, &A(j, p), lda, &B(p, 0), ldb, &C(j, 0), ldc);
        }
    }
}


void MY_MMult_Inner_Q_Pack(int m, int n, int k,  float *a, int lda, 
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
            InnerKernelQPack(jb, n, pb, &A(j, p), lda, &B(p, 0), ldb, &C(j, 0), ldc, j == 0);
        }
    }
}

void MY_MMult_Inner(int m, int n, int k,  float *a, int lda, 
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
            InnerKernel(jb, n, pb, &A(j, p), lda, &B(p, 0), ldb, &C(j, 0), ldc);
        }
    }
}




int main()
{
    int p, m, n, k,lda, ldb, ldc;
    float *a, *b, *c;    

    for (p = PFIRST; p <= PLAST; p += PINC )
    {
        m = p;
        n = p;
        k = p;
        lda = k;
        ldb = n;
        ldc = n;
        a = ( float * ) malloc( m * k * sizeof( float ) );  
        b = ( float * ) malloc( n * k * sizeof( float ) );
        c = ( float * ) malloc( m * n * sizeof( float ) );
        random_matrix(m, k, a, lda);
        random_matrix(k, n, b, ldb);  
        //printf("A:\r\n");
        //print_matrix(m, k, a, lda);
        //printf("B:\r\n");
        //print_matrix(k, n, b, ldb);

        struct timespec start, finish;
        double elapsed0 = 0,elapsed1 = 0, elapsed2=0
         ,elapsed3 = 0, elapsed4 = 0, elapsed5 = 0, elapsed6 = 0;
         
        clock_gettime(CLOCK_MONOTONIC, &start);
        native_c( m, n, k, a, lda, b, ldb, c, ldc );
        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed0 = (finish.tv_sec - start.tv_sec) * 1000000000;
        elapsed0 += (finish.tv_nsec - start.tv_nsec);
        float native_c_re_3311 =  c[11];
        //printf("C native:\r\n");
        //print_matrix(m, n, c, ldc);
        memset(c, 0 , m * n * sizeof(float));
        

        
        clock_gettime(CLOCK_MONOTONIC, &start);
        MY_MMult( m, n, k, a, lda, b, ldb, c, ldc );
        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed1 = (finish.tv_sec - start.tv_sec) * 1000000000;
        elapsed1 += (finish.tv_nsec - start.tv_nsec);
        float MY_MMult_3311 =  c[11];
        memset(c, 0 , m * n * sizeof(float));

        clock_gettime(CLOCK_MONOTONIC, &start);
        MY_MMult_k4v( m, n, k, a, lda, b, ldb, c, ldc );
        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed2 = (finish.tv_sec - start.tv_sec) * 1000000000;
        elapsed2 += (finish.tv_nsec - start.tv_nsec);
        float MY_MMult_4kv_3311 =  c[11];
        memset(c, 0 , m * n * sizeof(float));
        

        clock_gettime(CLOCK_MONOTONIC, &start);
        MY_MMult_k4v( m, n, k, a, lda, b, ldb, c, ldc );
        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed3 = (finish.tv_sec - start.tv_sec) * 1000000000;
        elapsed3 += (finish.tv_nsec - start.tv_nsec);
        float MY_MMult_4kvs_3311 =  c[11];
        memset(c, 0 , m * n * sizeof(float));

        clock_gettime(CLOCK_MONOTONIC, &start);
        MY_MMult_Inner( m, n, k, a, lda, b, ldb, c, ldc );
        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed4 = (finish.tv_sec - start.tv_sec) * 1000000000;
        elapsed4 += (finish.tv_nsec - start.tv_nsec);
        float MY_MMult_Inner_3311 =  c[11];
        memset(c, 0 , m * n * sizeof(float));

        clock_gettime(CLOCK_MONOTONIC, &start);
        MY_MMult_Inner_Pack( m, n, k, a, lda, b, ldb, c, ldc );
        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed5 = (finish.tv_sec - start.tv_sec) * 1000000000;
        elapsed5 += (finish.tv_nsec - start.tv_nsec);
        float Inner_PackA_11 =  c[11];
        memset(c, 0 , m * n * sizeof(float));

        clock_gettime(CLOCK_MONOTONIC, &start);
        MY_MMult_Inner_Q_Pack( m, n, k, a, lda, b, ldb, c, ldc );
        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed6 = (finish.tv_sec - start.tv_sec) * 1000000000;
        elapsed6 += (finish.tv_nsec - start.tv_nsec);
        float Inner_Q_PackA_11 =  c[11];
        memset(c, 0 , m * n * sizeof(float));
        
        printf("res:  %f %f %f %f\r\n",native_c_re_3311, MY_MMult_Inner_3311, Inner_PackA_11, Inner_Q_PackA_11);
        printf("time: %f %f %f %f %f %f %f\r\n",elapsed0,elapsed1,elapsed2,elapsed3,elapsed4,elapsed5,elapsed6);
        free(a);free(b);free(c);
    }
}

