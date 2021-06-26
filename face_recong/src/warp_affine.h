#ifndef WARP_AFFINE_C_H_
#define WARP_AFFINE_C_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define DL_IMAGE_MIN(A, B) ((A) < (B) ? (A) : (B))
#define DL_IMAGE_MAX(A, B) ((A) < (B) ? (B) : (A))

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

#define readfile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "rb");\
  if(out != NULL)\
  {\
        fread (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)
    
typedef float fptp_t;
typedef uint8_t uc_t;

typedef struct
{
    int w;        /*!< Width */
    int h;        /*!< Height */
    int c;        /*!< Channel */
    int n;        /*!< Number of filter, input and output must be 1 */
    int stride;   /*!< Step between lines */
    float *item; /*!< Data */
} dl_matrix3d_t;

typedef struct
{
    int w;      /*!< Width */
    int h;      /*!< Height */
    int c;      /*!< Channel */
    int n;      /*!< Number of filter, input and output must be 1 */
    int stride; /*!< Step between lines */
    uint8_t *item; /*!< Data */
} dl_matrix3du_t;

typedef float matrixType;
typedef struct
{
    int w;              /*!< width */
    int h;              /*!< height */
    matrixType **array; /*!< array */
} Matrix;

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


void warp_affine(dl_matrix3du_t *img, dl_matrix3du_t *crop, Matrix *M)
{
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
}

dl_matrix3d_t *get_face_id(dl_matrix3du_t *aligned_face)
{
    dl_matrix3d_t *face_id = NULL;
    //face_id从人脸识别模型导出来一个vector
    l2_norm(face_id);
    return face_id;
}


void image_cropper(uint8_t *rot_data, uint8_t *src_data, int rot_w, int rot_h, int rot_c, int src_w, int src_h, float rotate_angle, float ratio, float *center)
{ /*{{{*/
    int rot_stride = rot_w * rot_c;
    float rot_w_start = 0.5f - (float)rot_w / 2;
    float rot_h_start = 0.5f - (float)rot_h / 2;

    //rotate_angle must be radius
    float si = sin(rotate_angle);
    float co = cos(rotate_angle);

    int src_stride = src_w * rot_c;

    for (int y = 0; y < rot_h; y++)
    {
        for (int x = 0; x < rot_w; x++)
        {
            float xs, ys, xr, yr;
            xs = ratio * (rot_w_start + x);
            ys = ratio * (rot_h_start + y);

            xr = xs * co + ys * si;
            yr = -xs * si + ys * co;

            float fy[2];
            fy[0] = center[1] + yr; // y
            int src_y = (int)fy[0]; // y1
            fy[0] -= src_y;         // y - y1
            fy[1] = 1 - fy[0];      // y2 - y
            src_y = DL_IMAGE_MAX(0, src_y);
            src_y = DL_IMAGE_MIN(src_y, src_h - 2);

            float fx[2];
            fx[0] = center[0] + xr; // x
            int src_x = (int)fx[0]; // x1
            fx[0] -= src_x;         // x - x1
            if (src_x < 0)
            {
                fx[0] = 0;
                src_x = 0;
            }
            if (src_x > src_w - 2)
            {
                fx[0] = 0;
                src_x = src_w - 2;
            }
            fx[1] = 1 - fx[0]; // x2 - x

            for (int c = 0; c < rot_c; c++)
            {
                rot_data[y * rot_stride + x * rot_c + c] = round(src_data[src_y * src_stride + src_x * rot_c + c] * fx[1] * fy[1] + src_data[src_y * src_stride + (src_x + 1) * rot_c + c] * fx[0] * fy[1] + src_data[(src_y + 1) * src_stride + src_x * rot_c + c] * fx[1] * fy[0] + src_data[(src_y + 1) * src_stride + (src_x + 1) * rot_c + c] * fx[0] * fy[0]);
            }
        }
    }
} /*}}}*/

void image_resize_linear(uint8_t *dst_image, uint8_t *src_image, int dst_w, int dst_h, int dst_c, int src_w, int src_h)
{ /*{{{*/
    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;

    int dst_stride = dst_c * dst_w;
    int src_stride = dst_c * src_w;
    {
        for (int y = 0; y < dst_h; y++)
        {
            float fy[2];
            fy[0] = (float)((y + 0.5) * scale_y - 0.5); // y
            int src_y = (int)fy[0];                     // y1
            fy[0] -= src_y;                             // y - y1
            fy[1] = 1 - fy[0];                          // y2 - y
            src_y = DL_IMAGE_MAX(0, src_y);
            src_y = DL_IMAGE_MIN(src_y, src_h - 2);

            for (int x = 0; x < dst_w; x++)
            {
                float fx[2];
                fx[0] = (float)((x + 0.5) * scale_x - 0.5); // x
                int src_x = (int)fx[0];                     // x1
                fx[0] -= src_x;                             // x - x1
                if (src_x < 0)
                {
                    fx[0] = 0;
                    src_x = 0;
                }
                if (src_x > src_w - 2)
                {
                    fx[0] = 0;
                    src_x = src_w - 2;
                }
                fx[1] = 1 - fx[0]; // x2 - x

                for (int c = 0; c < dst_c; c++)
                {
                    dst_image[y * dst_stride + x * dst_c + c] = round(src_image[src_y * src_stride + src_x * dst_c + c] * fx[1] * fy[1] + src_image[src_y * src_stride + (src_x + 1) * dst_c + c] * fx[0] * fy[1] + src_image[(src_y + 1) * src_stride + src_x * dst_c + c] * fx[1] * fy[0] + src_image[(src_y + 1) * src_stride + (src_x + 1) * dst_c + c] * fx[0] * fy[0]);
                }
            }
        }
    }
} /*}}}*/

#endif
