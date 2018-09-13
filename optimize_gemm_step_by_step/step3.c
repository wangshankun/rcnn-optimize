//格式与numpy一致i是列的index; j是行的index
#define A(j, i) a[ (j)*lda + (i) ]
#define B(j, i) b[ (j)*ldb + (i) ]
#define C(j, i) c[ (j)*ldc + (i) ]

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