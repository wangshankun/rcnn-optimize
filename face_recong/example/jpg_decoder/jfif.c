#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "stdefine.h"
#include "bitstr.h"
#include "huffman.h"
#include "quant.h"
#include "zigzag.h"
#include "dct.h"
#include "color.h"
#include "jfif.h"

#define DEBUG_JFIF  0

typedef struct {
    // width & height
    int       width;
    int       height;

    // quantization table
    int      *pqtab[16];

    // huffman codec ac
    HUFCODEC *phcac[16];

    // huffman codec dc
    HUFCODEC *phcdc[16];

    // components
    int comp_num;
    struct {
        int id;
        int samp_factor_v;
        int samp_factor_h;
        int qtab_idx;
        int htab_idx_ac;
        int htab_idx_dc;
    } comp_info[4];

    int   datalen;
    BYTE *databuf;
} JFIF;

#if DEBUG_JFIF
static void jfif_dump(JFIF *jfif)
{
    int i, j;

    printf("++ jfif dump ++\n");
    printf("width : %d\n", jfif->width );
    printf("height: %d\n", jfif->height);
    printf("\n");

    for (i=0; i<16; i++) {
        if (!jfif->pqtab[i]) continue;
        printf("qtab%d\n", i);
        for (j=0; j<64; j++) {
            printf("%3d,%c", jfif->pqtab[i][j], j%8 == 7 ? '\n' : ' ');
        }
        printf("\n");
    }

    for (i=0; i<16; i++) {
        int size = 16;
        if (!jfif->phcac[i]) continue;
        printf("htabac%d\n", i);
        for (j=0; j<16; j++) {
            size += jfif->phcac[i]->huftab[j];
        }
        for (j=0; j<size; j++) {
            printf("%3d,%c", jfif->phcac[i]->huftab[j], j%16 == 15 ? '\n' : ' ');
        }
        printf("\n\n");
    }

    for (i=0; i<16; i++) {
        int size = 16;
        if (!jfif->phcdc[i]) continue;
        printf("htabdc%d\n", i);
        for (j=0; j<16; j++) {
            size += jfif->phcdc[i]->huftab[j];
        }
        for (j=0; j<size; j++) {
            printf("%3d,%c", jfif->phcdc[i]->huftab[j], j%16 == 15 ? '\n' : ' ');
        }
        printf("\n\n");
    }

    printf("comp_num : %d\n", jfif->comp_num);
    for (i=0; i<jfif->comp_num; i++) {
        printf("id:%d samp_factor_v:%d samp_factor_h:%d qtab_idx:%d htab_idx_ac:%d htab_idx_dc:%d\n",
            jfif->comp_info[i].id,
            jfif->comp_info[i].samp_factor_v,
            jfif->comp_info[i].samp_factor_h,
            jfif->comp_info[i].qtab_idx,
            jfif->comp_info[i].htab_idx_ac,
            jfif->comp_info[i].htab_idx_dc);
    }
    printf("\n");

    printf("datalen : %d\n", jfif->datalen);
    printf("-- jfif dump --\n");
}

static void dump_du(int *du)
{
    int i;
    for (i=0; i<64; i++) {
        printf("%3d%c", du[i], i % 8 == 7 ? '\n' : ' ');
    }
    printf("\n");
}
#endif

static int category_decode(int code, int  size)
{
    return code >= (1 << (size - 1)) ? code : code - (1 << size) + 1;
}

static void jfif_free(void *ctxt)
{
    JFIF *jfif = (JFIF*)ctxt;
    int   i;
    if (!jfif) return;
    for (i=0; i<16; i++) {
        if (jfif->pqtab[i]) free(jfif->pqtab[i]);
        if (jfif->phcac[i]) free(jfif->phcac[i]);
        if (jfif->phcdc[i]) free(jfif->phcdc[i]);
    }
    if (jfif->databuf) free(jfif->databuf);
    free(jfif);
}

static void* jfif_load(const char *file)
{
    JFIF *jfif   = NULL;
    FILE *fp     = NULL;
    int   header = 0;
    int   type   = 0;
    WORD  size   = 0;
    BYTE *buf    = NULL;
    BYTE *end    = NULL;
    BYTE *dqt, *dht;
    int   ret    =-1;
    long  offset = 0;
    int   i;
    jfif = calloc(1, sizeof(JFIF));
    buf  = calloc(1, 0x10000);
    end  = buf + 0x10000;
    if (!jfif || !buf) goto done;
    fp = fopen(file, "rb");
    if (!fp) 
    {
        perror("Error");
        goto done;
    }
    while (1) {
        do { header = fgetc(fp); } while (header != EOF && header != 0xff); // get header
        do { type   = fgetc(fp); } while (type   != EOF && type   == 0xff); // get type
        if (header == EOF || type == EOF) {
            printf("file eof !\n");
            break;
        }

        if ((type == 0xd8) || (type == 0xd9) || (type == 0x01) || (type >= 0xd0 && type <= 0xd7)) {
            size = 0;
        } else {
            size  = fgetc(fp) << 8;
            size |= fgetc(fp) << 0;
            size -= 2;
        }

        size = fread(buf, 1, size, fp);
        switch (type) {
        case 0xc0: // SOF0
            jfif->width    = (buf[3] << 8) | (buf[4] << 0);
            jfif->height   = (buf[1] << 8) | (buf[2] << 0);
            jfif->comp_num =  buf[5] < 4 ? buf[5] : 4;
            for (i=0; i<jfif->comp_num; i++) {
                jfif->comp_info[i].id = buf[6 + i * 3];
                jfif->comp_info[i].samp_factor_v = (buf[7 + i * 3] >> 0) & 0x0f;
                jfif->comp_info[i].samp_factor_h = (buf[7 + i * 3] >> 4) & 0x0f;
                jfif->comp_info[i].qtab_idx      =  buf[8 + i * 3] & 0x0f;
            }
            break;

        case 0xda: // SOS
            jfif->comp_num = buf[0] < 4 ? buf[0] : 4;
            for (i=0; i<jfif->comp_num; i++) {
                jfif->comp_info[i].id = buf[1 + i * 2];
                jfif->comp_info[i].htab_idx_ac = (buf[2 + i * 2] >> 0) & 0x0f;
                jfif->comp_info[i].htab_idx_dc = (buf[2 + i * 2] >> 4) & 0x0f;
            }
            offset = ftell(fp);
            ret    = 0;
            goto read_data;

        case 0xdb: // DQT
            dqt = buf;
            while (size > 0 && dqt < end) {
                int idx = dqt[0] & 0x0f;
                int f16 = dqt[0] & 0xf0;
                if (!jfif->pqtab[idx]) jfif->pqtab[idx] = malloc(64 * sizeof(int));
                if (!jfif->pqtab[idx]) break;
                if (dqt + 1 + 64 + (f16 ? 64 : 0) < end) {
                    for (i=0; i<64; i++) {
                        jfif->pqtab[idx][ZIGZAG[i]] = f16 ? ((dqt[1 + i * 2] << 8) | (dqt[2 + i * 2] << 0)) : dqt[1 + i];
                    }
                }
                dqt += 1 + 64 + (f16 ? 64 : 0);
                size-= 1 + 64 + (f16 ? 64 : 0);
            }
            break;

        case 0xc4: // DHT
            dht = buf;
            while (size > 0 && dht + 17 < end) {
                int idx = dht[0] & 0x0f;
                int fac = dht[0] & 0xf0;
                int len = 0;
                for (i=1; i<1+16; i++) len += dht[i];
                if (len > end - dht - 17) len = end - dht - 17;
                if (len > 256) len = 256;
                if (fac) {
                    if (!jfif->phcac[idx]) jfif->phcac[idx] = calloc(1, sizeof(HUFCODEC));
                    if ( jfif->phcac[idx]) memcpy(jfif->phcac[idx]->huftab, &dht[1], 16 + len);
                } else {
                    if (!jfif->phcdc[idx]) jfif->phcdc[idx] = calloc(1, sizeof(HUFCODEC));
                    if ( jfif->phcdc[idx]) memcpy(jfif->phcdc[idx]->huftab, &dht[1], 16 + len);
                }
                dht += 17 + len;
                size-= 17 + len;
            }
            break;
        }
    }

read_data:
    fseek(fp, 0, SEEK_END);
    jfif->datalen = ftell(fp) - offset;
    jfif->databuf = malloc(jfif->datalen);
    if (jfif->databuf) {
        fseek(fp, offset, SEEK_SET);
        fread(jfif->databuf, 1, jfif->datalen, fp);
    }

done:
    if (buf) free  (buf);
    if (fp ) fclose(fp );
    if (ret == -1) {
        jfif_free(jfif);
        jfif = NULL;
    }
    return jfif;
}

#define DU_TYPE_LUMIN  0
#define DU_TYPE_CHROM  1

typedef struct {
    unsigned runlen   : 4;
    unsigned codesize : 4;
    unsigned codedata : 16;
} RLEITEM;

static int jfif_decode_rgb(void *ctxt, uint8_t **data, int *width, int *height)
{
    JFIF *jfif    = (JFIF*)ctxt;
    void *bs      = NULL;
    int  *ftab[16]= {0};
    int   dc[4]   = {0};
    int   mcuw, mcuh, mcuc, mcur, mcui, jw, jh;
    int   i, j, c, h, v, x, y;
    int   sfh_max = 0;
    int   sfv_max = 0;
    int   yuv_stride[4] = {0};
    int   yuv_height[4] = {0};
    int  *yuv_datbuf[4] = {0};
    int  *idst, *isrc;
    int  *ysrc, *usrc, *vsrc;
    int   ret = -1;

    if (!ctxt) {
        printf("invalid input params !\n");
        return -1;
    }

    // init dct module
    init_dct_module();

    //++ init ftab
    for (i=0; i<16; i++) {
        if (jfif->pqtab[i]) {
            ftab[i] = malloc(64 * sizeof(int));
            if (ftab[i]) {
                init_idct_ftab(ftab[i], jfif->pqtab[i]);
            } else {
                goto done;
            }
        }
    }
    //-- init ftab

    //++ calculate mcu info
    for (c=0; c<jfif->comp_num; c++) {
        if (sfh_max < jfif->comp_info[c].samp_factor_h) {
            sfh_max = jfif->comp_info[c].samp_factor_h;
        }
        if (sfv_max < jfif->comp_info[c].samp_factor_v) {
            sfv_max = jfif->comp_info[c].samp_factor_v;
        }
    }
    if (!sfh_max) sfh_max = 1;
    if (!sfv_max) sfv_max = 1;
    mcuw = sfh_max * 8;
    mcuh = sfv_max * 8;
    jw = ALIGN(jfif->width , mcuw);
    jh = ALIGN(jfif->height, mcuh);
    mcuc = jw / mcuw;
    mcur = jh / mcuh;
    //-- calculate mcu info

    // create yuv buffer for decoding
    yuv_stride[0] = jw;
    yuv_stride[1] = jw * jfif->comp_info[1].samp_factor_h / sfh_max;
    yuv_stride[2] = jw * jfif->comp_info[2].samp_factor_h / sfh_max;
    yuv_stride[3] = jw * jfif->comp_info[3].samp_factor_h / sfh_max;
    yuv_height[0] = jh;
    yuv_height[1] = jh * jfif->comp_info[1].samp_factor_v / sfv_max;
    yuv_height[2] = jh * jfif->comp_info[2].samp_factor_v / sfv_max;
    yuv_height[3] = jh * jfif->comp_info[3].samp_factor_v / sfv_max;
    yuv_datbuf[0] = malloc(sizeof(int) * yuv_stride[0] * yuv_height[0]);
    yuv_datbuf[1] = malloc(sizeof(int) * yuv_stride[1] * yuv_height[1]);
    yuv_datbuf[2] = malloc(sizeof(int) * yuv_stride[2] * yuv_height[2]);
    yuv_datbuf[3] = malloc(sizeof(int) * yuv_stride[3] * yuv_height[3]);
    if (!yuv_datbuf[0] || !yuv_datbuf[1] || !yuv_datbuf[2] || !yuv_datbuf[3]) {
        goto done;
    }

    // open bit stream
    bs = bitstr_open(jfif->databuf, "mem", jfif->datalen);
    if (!bs) {
        printf("failed to open bitstr for jfif_decode !");
        return -1;
    }

    // init huffman codec
    for (i=0; i<16; i++) {
        if (jfif->phcac[i]) {
            jfif->phcac[i]->input = bs;
            huffman_decode_init(jfif->phcac[i]);
        }
        if (jfif->phcdc[i]) {
            jfif->phcdc[i]->input = bs;
            huffman_decode_init(jfif->phcdc[i]);
        }
    }

    for (mcui=0; mcui<mcuc*mcur; mcui++) {
        for (c=0; c<jfif->comp_num; c++) {
            for (v=0; v<jfif->comp_info[c].samp_factor_v; v++) {
                for (h=0; h<jfif->comp_info[c].samp_factor_h; h++) {
                    HUFCODEC *hcac = jfif->phcac[jfif->comp_info[c].htab_idx_ac];
                    HUFCODEC *hcdc = jfif->phcdc[jfif->comp_info[c].htab_idx_dc];
                    int       fidx = jfif->comp_info[c].qtab_idx;
                    int size, znum, code;
                    int du[64] = {0};

                    //+ decode dc
                    size = huffman_decode_step(hcdc) & 0xf;
                    if (size) {
                        code = bitstr_get_bits(bs  , size);
                        code = category_decode(code, size);
                    }
                    else {
                        code = 0;
                    }
                    dc[c] += code;
                    du[0]  = dc[c];
                    //- decode dc

                    //+ decode ac
                    for (i=1; i<64; ) {
                        code = huffman_decode_step(hcac);
                        if (code <= 0) break;
                        size = (code >> 0) & 0xf;
                        znum = (code >> 4) & 0xf;
                        i   += znum;
                        code = bitstr_get_bits(bs  , size);
                        code = category_decode(code, size);
                        if (i < 64) du[i++] = code;
                    }
                    //- decode ac

                    // de-zigzag
                    zigzag_decode(du);

                    // idct
                    idct2d8x8(du, ftab[fidx]);

                    // copy du to yuv buffer
                    x    = ((mcui % mcuc) * mcuw + h * 8) * jfif->comp_info[c].samp_factor_h / sfh_max;
                    y    = ((mcui / mcuc) * mcuh + v * 8) * jfif->comp_info[c].samp_factor_v / sfv_max;
                    idst = yuv_datbuf[c] + y * yuv_stride[c] + x;
                    isrc = du;
                    for (i=0; i<8; i++) {
                        memcpy(idst, isrc, 8 * sizeof(int));
                        idst += yuv_stride[c];
                        isrc += 8;
                    }
                }
            }
        }
    }

    // close huffman codec
    for (i=0; i<16; i++) {
        if (jfif->phcac[i]) huffman_decode_done(jfif->phcac[i]);
        if (jfif->phcdc[i]) huffman_decode_done(jfif->phcdc[i]);
    }

    // close bit stream
    bitstr_close(bs);

 
    ysrc = yuv_datbuf[0];
    *width = jfif->width;
    *height = jfif->height;
    *data = (BYTE*)malloc(jfif->width * jfif->height * 3);

    BYTE* bdst = *data;
    for (i=0; i<jfif->height; i++) {
        int uy = i * jfif->comp_info[1].samp_factor_v / sfv_max;
        int vy = i * jfif->comp_info[2].samp_factor_v / sfv_max;
        for (j=0; j<jfif->width; j++) {
            int ux = j * jfif->comp_info[1].samp_factor_h / sfh_max;
            int vx = j * jfif->comp_info[2].samp_factor_h / sfh_max;
            usrc = yuv_datbuf[1] + uy * yuv_stride[1] + ux;
            vsrc = yuv_datbuf[2] + vy * yuv_stride[2] + vx;
            yuv_to_rgb(*ysrc, *vsrc, *usrc, bdst + 2, bdst + 1, bdst + 0);
            bdst += 3;
            ysrc += 1;
        }
        ysrc -= jfif->width * 1;
        ysrc += yuv_stride[0];
    }

    // success
    ret = 0;

done:
    if (yuv_datbuf[0]) free(yuv_datbuf[0]);
    if (yuv_datbuf[1]) free(yuv_datbuf[1]);
    if (yuv_datbuf[2]) free(yuv_datbuf[2]);
    if (yuv_datbuf[3]) free(yuv_datbuf[3]);
    //++ free ftab
    for (i=0; i<16; i++) {
        if (ftab[i]) {
            free(ftab[i]);
        }
    }
    //-- free ftab
    return ret;
}

int jpg_decode_rgb(const char *file, uint8_t **data, int *w, int *h)
{
    void *jfif = jfif_load(file);
    if (0 != jfif_decode_rgb(jfif, data, w, h))
    {
        jfif_free(jfif);
        return -1;
    }
    jfif_free(jfif);
    return 0;
}