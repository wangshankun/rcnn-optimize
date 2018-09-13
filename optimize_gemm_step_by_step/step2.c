//格式与numpy一致i是列的index; j是行的index
#define A(j, i) a[ (j)*lda + (i) ]
#define B(j, i) b[ (j)*ldb + (i) ]
#define C(j, i) c[ (j)*ldc + (i) ]

void AddDot1x1( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
    int p;

    for ( p = 0; p < k; p++ )
    {
        C( 0, 0 ) += A( 0, p ) * B( p, 0 ); 
    }
}

void MY_MMult( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
    int i, j;

    for (j = 0; j < m; j++)
    {
        for (i = 0; i < n; i++)
        {   //A每次出一行的行首，B每次出一列的列首
            AddDot1x1(k, &A(j, 0 ), lda, &B(0, i), ldb, &C(j, i), ldc );
        }
    }
}