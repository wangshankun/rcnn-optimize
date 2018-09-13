//格式与numpy一致i是列的index; j是行的index
#define A(j, i) a[ (j)*lda + (i) ]
#define B(j, i) b[ (j)*ldb + (i) ]
#define C(j, i) c[ (j)*ldc + (i) ]

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