#ifndef WARP_AFFINE_C_H_
#define WARP_AFFINE_C_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define DL_IMAGE_MIN(A, B) ((A) < (B) ? (A) : (B))
#define DL_IMAGE_MAX(A, B) ((A) < (B) ? (B) : (A))

typedef float fptp_t;
typedef uint8_t uc_t;

typedef struct
{
    int w = 0;        /*!< Width */
    int h = 0;        /*!< Height */
    int c = 0;        /*!< Channel */
    int n = 0;        /*!< Number of filter, input and output must be 1 */
    int stride = 0;   /*!< Step between lines */
    float *item = NULL; /*!< Data */
} dl_matrix3d_t;

typedef struct
{
    int w = 0;      /*!< Width */
    int h = 0;      /*!< Height */
    int c = 0;      /*!< Channel */
    int n = 0;      /*!< Number of filter, input and output must be 1 */
    int stride = 0; /*!< Step between lines */
    uint8_t *item = NULL; /*!< Data */
} dl_matrix3du_t;

typedef float matrixType;
typedef struct
{
    int w = 0;              /*!< width */
    int h = 0;              /*!< height */
    matrixType **array = NULL; /*!< array */
} Matrix;


Matrix *matrix_alloc(int h, int w)
{
    Matrix *r = (Matrix *)calloc(1, sizeof(Matrix));
    r->w = w;
    r->h = h;
    r->array = (matrixType**)calloc(h, sizeof(matrixType *));
    for (int i = 0; i < h; i++)
    {
        r->array[i] = (matrixType*)calloc(w, sizeof(matrixType));
    }
    return r;
}

void matrix_free(Matrix *m)
{
    for (int i = 0; i < m->h; i++)
    {
        free(m->array[i]);
    }
    free(m->array);
    free(m);
    //m = NULL;
}

void matrix_print(Matrix *m)
{
    printf("Matrix: %dx%d\n", m->h, m->w);
    for (int i = 0; i < m->h; i++)
    {
        for (int j = 0; j < m->w; j++)
        {
            printf("%f ", m->array[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

Matrix *get_similarity_matrix(float *srcx, float *srcy, float *dstx, float *dsty, int num)
{
    int dim = 2;
    double src_mean_x = 0.0;
    double src_mean_y = 0.0;
    double dst_mean_x = 0.0;
    double dst_mean_y = 0.0;

    for (int i = 0; i < num; i++)
    {
        src_mean_x += srcx[i];
        src_mean_y += srcy[i];
        dst_mean_x += dstx[i];
        dst_mean_y += dsty[i];
    }
    src_mean_x /= num;
    src_mean_y /= num;
    dst_mean_x /= num;
    dst_mean_y /= num;

    Matrix *src_demean = matrix_alloc(num, 2);
    Matrix *dst_demean = matrix_alloc(num, 2);
    for (int i = 0; i < num; i++)
    {
        src_demean->array[i][0] = srcx[i] - src_mean_x;
        src_demean->array[i][1] = srcy[i] - src_mean_y;
        dst_demean->array[i][0] = dstx[i] - dst_mean_x;
        dst_demean->array[i][1] = dsty[i] - dst_mean_y;
    }
    double A[2][2] = {0};
    for (int i = 0; i < num; i++)
    {
        A[0][0] += (dst_demean->array[i][0] * src_demean->array[i][0] / num);
        A[0][1] += (dst_demean->array[i][0] * src_demean->array[i][1] / num);
        A[1][0] += (dst_demean->array[i][1] * src_demean->array[i][0] / num);
        A[1][1] += (dst_demean->array[i][1] * src_demean->array[i][1] / num);
    }
    if ((A[0][0] == 0) && (A[0][1] == 0) && (A[1][0] == 0) && (A[1][1] == 0))
    {
        matrix_free(src_demean);
        matrix_free(dst_demean);
        return NULL;
    }

    double d[2] = {1, 1};
    if (((A[0][0] * A[1][1]) - A[0][1] * A[1][0]) < 0)
    {
        d[1] = -1;
    }

    //======================================================================SVD=====================================================================
    double U[2][2] = {0};
    double V[2][2] = {0};
    double S[2] = {0};

    double divide_temp = 0;

    double AAT[2][2] = {0};
    AAT[0][0] = A[0][0] * A[0][0] + A[0][1] * A[0][1];
    AAT[0][1] = A[0][0] * A[1][0] + A[0][1] * A[1][1];
    AAT[1][0] = A[1][0] * A[0][0] + A[1][1] * A[0][1];
    AAT[1][1] = A[1][0] * A[1][0] + A[1][1] * A[1][1];

    double l1 = (AAT[0][0] + AAT[1][1] + sqrt((AAT[0][0] + AAT[1][1]) * (AAT[0][0] + AAT[1][1]) - 4 * ((AAT[0][0] * AAT[1][1]) - (AAT[0][1] * AAT[1][0])))) / 2.0;
    double l2 = (AAT[0][0] + AAT[1][1] - sqrt((AAT[0][0] + AAT[1][1]) * (AAT[0][0] + AAT[1][1]) - 4 * ((AAT[0][0] * AAT[1][1]) - (AAT[0][1] * AAT[1][0])))) / 2.0;
    S[0] = sqrt(l1);
    S[1] = sqrt(l2);

    U[0][0] = 1.0;
    divide_temp = l1 - AAT[1][1];
    if (divide_temp == 0)
    {
        return NULL;
    }
    U[1][0] = AAT[1][0] / divide_temp;
    double norm = sqrt((U[0][0] * U[0][0]) + (U[1][0] * U[1][0]));
    U[0][0] /= norm;
    U[1][0] /= norm;

    U[0][1] = 1.0;
    divide_temp = l2 - AAT[1][1];
    if (divide_temp == 0)
    {
        return NULL;
    }
    U[1][1] = AAT[1][0] / divide_temp;
    norm = sqrt((U[0][1] * U[0][1]) + (U[1][1] * U[1][1]));
    U[0][1] /= norm;
    U[1][1] /= norm;

    if (U[0][1] * U[1][0] < 0)
    {
        U[0][0] = -U[0][0];
        U[1][0] = -U[1][0];
    }

    double ATA[2][2] = {0};
    ATA[0][0] = A[0][0] * A[0][0] + A[1][0] * A[1][0];
    ATA[0][1] = A[0][0] * A[0][1] + A[1][0] * A[1][1];
    ATA[1][0] = A[0][1] * A[0][0] + A[1][1] * A[1][0];
    ATA[1][1] = A[0][1] * A[0][1] + A[1][1] * A[1][1];

    V[0][0] = 1.0;
    divide_temp = l1 - ATA[1][1];
    if (divide_temp == 0)
    {
        return NULL;
    }
    V[0][1] = ATA[1][0] / divide_temp;
    norm = sqrt((V[0][0] * V[0][0]) + (V[0][1] * V[0][1]));
    V[0][0] /= norm;
    V[0][1] /= norm;

    V[1][0] = 1.0;
    divide_temp = l2 - ATA[1][1];
    if (divide_temp == 0)
    {
        return NULL;
    }
    V[1][1] = ATA[1][0] / divide_temp;
    norm = sqrt((V[1][0] * V[1][0]) + (V[1][1] * V[1][1]));
    V[1][0] /= norm;
    V[1][1] /= norm;

    if (V[0][1] * V[1][0] < 0)
    {
        V[0][0] = -V[0][0];
        V[0][1] = -V[0][1];
    }
    if ((S[0] * U[0][0] * V[0][0] + S[1] * U[0][1] * V[1][0]) * A[0][0] < 0)
    {
        U[0][0] = -U[0][0];
        U[0][1] = -U[0][1];
        U[1][0] = -U[1][0];
        U[1][1] = -U[1][1];
    }
    //============================================================================================================================================

    Matrix *T = matrix_alloc(2, 3);
    if (fabs((A[0][0] * A[1][1]) - A[0][1] * A[1][0]) < 1e-8)
    {
        if ((((U[0][0] * U[1][1]) - U[0][1] * U[1][0]) * ((V[0][0] * V[1][1]) - V[0][1] * V[1][0])) > 0)
        {
            T->array[0][0] = U[0][0] * V[0][0] + U[0][1] * V[1][0];
            T->array[0][1] = U[0][0] * V[0][1] + U[0][1] * V[1][1];
            T->array[1][0] = U[1][0] * V[0][0] + U[1][1] * V[1][0];
            T->array[1][1] = U[1][0] * V[0][1] + U[1][1] * V[1][1];
        }
        else
        {
            double s = d[dim - 1];
            d[dim - 1] = -1;
            T->array[0][0] = d[0] * U[0][0] * V[0][0] + d[1] * U[0][1] * V[1][0];
            T->array[0][1] = d[0] * U[0][0] * V[0][1] + d[1] * U[0][1] * V[1][1];
            T->array[1][0] = d[0] * U[1][0] * V[0][0] + d[1] * U[1][1] * V[1][0];
            T->array[1][1] = d[0] * U[1][0] * V[0][1] + d[1] * U[1][1] * V[1][1];
            d[dim - 1] = s;
        }
    }
    else
    {
        T->array[0][0] = d[0] * U[0][0] * V[0][0] + d[1] * U[0][1] * V[1][0];
        T->array[0][1] = d[0] * U[0][0] * V[0][1] + d[1] * U[0][1] * V[1][1];
        T->array[1][0] = d[0] * U[1][0] * V[0][0] + d[1] * U[1][1] * V[1][0];
        T->array[1][1] = d[0] * U[1][0] * V[0][1] + d[1] * U[1][1] * V[1][1];
    }

    double Ex = 0.0;
    double Ex2 = 0.0;
    double Ey = 0.0;
    double Ey2 = 0.0;
    for (int i = 0; i < num; i++)
    {
        Ex += src_demean->array[i][0];
        Ex2 += (src_demean->array[i][0] * src_demean->array[i][0]);
        Ey += src_demean->array[i][1];
        Ey2 += (src_demean->array[i][1] * src_demean->array[i][1]);
    }
    Ex /= num;
    Ex2 /= num;
    Ey /= num;
    Ey2 /= num;
    double var_sum = (Ex2 - Ex * Ex) + (Ey2 - Ey * Ey);
    double scale = (S[0] * d[0] + S[1] * d[1]) / var_sum;

    T->array[0][2] = dst_mean_x - scale * (T->array[0][0] * src_mean_x + T->array[0][1] * src_mean_y);
    T->array[1][2] = dst_mean_y - scale * (T->array[1][0] * src_mean_x + T->array[1][1] * src_mean_y);

    T->array[0][0] *= scale;
    T->array[0][1] *= scale;
    T->array[1][0] *= scale;
    T->array[1][1] *= scale;

    matrix_free(src_demean);
    matrix_free(dst_demean);
    return T;
}

Matrix *get_inv_affine_matrix(Matrix *m)
{
    Matrix *minv = matrix_alloc(2, 3);
    float mdet = (m->array[0][0]) * (m->array[1][1]) - (m->array[1][0]) * (m->array[0][1]);
    if (mdet == 0)
    {
        printf("the matrix m is wrong !\n");
        return NULL;
    }

    minv->array[0][0] = m->array[1][1] / mdet;
    minv->array[0][1] = -(m->array[0][1] / mdet);
    minv->array[0][2] = ((m->array[0][1]) * (m->array[1][2]) - (m->array[0][2]) * (m->array[1][1])) / mdet;
    minv->array[1][0] = -(m->array[1][0]) / mdet;
    minv->array[1][1] = (m->array[0][0]) / mdet;
    minv->array[1][2] = ((m->array[0][2]) * (m->array[1][0]) - (m->array[0][0]) * (m->array[1][2])) / mdet;
    return minv;
}


int warp_affine(dl_matrix3du_t *img, dl_matrix3du_t *crop, Matrix *M)
{
    int pad_count = 0;
    Matrix *M_inv = get_inv_affine_matrix(M);
    uint8_t *dst = crop->item;
    int stride = img->w * img->c;
    int c = img->c;
    float x_src = 0.0;
    float y_src = 0.0;
    int x1 = 0;
    int x2 = 0;
    int y1 = 0;
    int y2 = 0;
    for (int i = 0; i < crop->h; i++)
    {
        for (int j = 0; j < crop->w; j++)
        {
            x_src = M_inv->array[0][0] * j + M_inv->array[0][1] * i + M_inv->array[0][2];
            y_src = M_inv->array[1][0] * j + M_inv->array[1][1] * i + M_inv->array[1][2];
            if ((x_src < 0) || (y_src < 0) || (x_src >= (img->w - 1)) || (y_src >= (img->h - 1)))
            {
                for (int k = 0; k < crop->c; k++)
                {
                    *dst++ = 0;
                    //*dst++ = 128;
                    pad_count++;
                }
            }
            else
            {
                x1 = floor(x_src);
                x2 = x1 + 1;
                y1 = floor(y_src);
                y2 = y1 + 1;
                for (int k = 0; k < crop->c; k++)
                {
                    *dst++ = (uint8_t)rintf(((img->item[y1 * stride + x1 * c + k]) * (x2 - x_src) * (y2 - y_src)) + ((img->item[y1 * stride + x2 * c + k]) * (x_src - x1) * (y2 - y_src)) + ((img->item[y2 * stride + x1 * c + k]) * (x2 - x_src) * (y_src - y1)) + ((img->item[y2 * stride + x2 * c + k]) * (x_src - x1) * (y_src - y1)));
                }
            }
        }
    }
    matrix_free(M_inv);
    return pad_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void l2_norm(dl_matrix3d_t *feature)
{
    int len = feature->w * feature->h * feature->c;
    fptp_t norm = 0;
    for(int i=0;i<len;i++){
        norm += (feature->item[i] * feature->item[i]);
    }
    norm = sqrt(norm);
    for(int i=0;i<len;i++){
        feature->item[i] /= norm;
    }
}

//已经做过l2_norm可以用cos_distance_unit_id计算余弦距离
fptp_t cos_distance_unit_id(dl_matrix3d_t *id_1,
                    dl_matrix3d_t *id_2)
{
    uint16_t c = id_1->c;
    fptp_t dist = 0;
    for (uint16_t i = 0; i < c; i++)
    {
        dist += ((id_1->item[i]) * (id_2->item[i]));
    }
    return dist;
}

//直接计算两mat的余弦距离
fptp_t cos_distance(dl_matrix3d_t *id_1,
                    dl_matrix3d_t *id_2)
{
    uint16_t c = id_1->c;
    fptp_t l2_norm_1 = 0;
    fptp_t l2_norm_2 = 0;
    fptp_t dist = 0;
    for (int i = 0; i < c; i++)
    {
        l2_norm_1 += ((id_1->item[i]) * (id_1->item[i]));
        l2_norm_2 += ((id_2->item[i]) * (id_2->item[i]));
    }
    l2_norm_1 = sqrt(l2_norm_1);
    l2_norm_2 = sqrt(l2_norm_2);
    for (uint16_t i = 0; i < c; i++)
    {
        dist += ((id_1->item[i]) * (id_2->item[i]) / (l2_norm_1 * l2_norm_2));
    }
    return dist;
}



#endif
