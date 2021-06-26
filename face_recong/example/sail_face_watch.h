#ifndef SAIL_FACE_WATCH_C_H_
#define SAIL_FACE_WATCH_C_H_

#if defined(__cplusplus)
extern "C" {

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
    unsigned char *item; /*!< Data */
} dl_matrix3du_t;

int face_det(dl_matrix3du_t* org_img, dl_matrix3du_t* det_face_img);
int face_align(dl_matrix3du_t* det_face_img, dl_matrix3du_t* face_align_img);
int face_recong(dl_matrix3du_t* face_align_img, dl_matrix3d_t* face_vector);

}
#endif//defined(__cplusplus)
#endif//SAIL_FACE_WATCH_C_H_