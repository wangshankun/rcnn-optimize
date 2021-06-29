#include "sail_face_watch.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <string.h>
#include <vector>
#include<vector>

using namespace std;
#define savefile(name_x_, buffer_x_, size_x_) do\
{\
  FILE *fp_out_x_ = fopen(name_x_, "wb");\
  if(fp_out_x_ != NULL)\
  {\
        fwrite (buffer_x_ , sizeof(char), size_x_, fp_out_x_);\
        fclose (fp_out_x_);\
  }\
} while(0)

#define readfile(name_x_, buffer_x_, size_x_) do\
{\
  FILE *fp_out_x_ = fopen(name_x_, "rb");\
  if(fp_out_x_ != NULL)\
  {\
        fread (buffer_x_ , sizeof(char), size_x_, fp_out_x_);\
        fclose (fp_out_x_);\
  }\
} while(0)


static vector<string> split(const string& str, const string& delim) {
    vector<string> res;
    if("" == str) return res;

    char * strs = new char[str.length() + 1] ; 
    strcpy(strs, str.c_str()); 
 
    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());
 
    char *p = strtok(strs, d);
    while(p) {
        string s = p;
        res.push_back(s);
        p = strtok(NULL, d);
    }
 
    return res;
}

void rgb2gray(dl_matrix3du_t *img, dl_matrix3du_t *gray)
{
    assert(img->c == 3);
    int count = (img->w) * (img->h);
    uint8_t *r = img->item;
    uint8_t *g = r + 1;
    uint8_t *b = r + 2;

    uint8_t *pgray = gray->item;
    int x = 0;
    for (int i = 0; i < count; i++)
    {
        x = (19595 * (*r) + 38469 * (*g) + 7472 * (*b)) >> 16; //fast algorithm
        *(pgray++) = (uint8_t)x;
        r += 3;
        g += 3;
        b += 3;
    }
}

#include "jfif.h"
#include <fstream>
#include <iostream>
using namespace std;
int main(int argc, const char* argv[])
{
    char *pathvar_c = NULL;
    pathvar_c = getenv("WATCH_TEST_PATH");
    if(pathvar_c == NULL)
    {
        printf("Need to set env var: WATCH_TEST_PATH !\r\n");
        return -1;
    }

    if(argc < 2)
    {
	printf("./test test_file_list.txt\r\n");
	return -1;
    }
    string pathvar(pathvar_c);
    pathvar = pathvar + "/";
    string db_path = pathvar + "test.bin";
    FaceHandle_t face_handle;
    face_handle.db_path = (char*)db_path.c_str();
    face_handle.reg_th_hold = 0.60;
    face_handle.det_th_hold = 0.35;
    if(Init(&face_handle) != 0)
    {
        return -1;
    }

    ifstream fin; string line;
    //入库
    string  input_db_file_list = pathvar + "base_clean.txt";
    fin.open(input_db_file_list);
    while (getline(fin, line)) 
    {
        //string img_path = pathvar + line;
        string img_path = line;
        uint8_t* data;
        int w = 0;
        int h = 0;
        printf("%s \r\n",img_path.c_str());
        if( 0 == jpg_decode_rgb(img_path.c_str(), &data, &w, &h))
        {
            if ( w == 0 || h == 0 )
            {
                printf("jpg_decode_rgb w or h error!\r\n");
                return -1;
            }
        }
        else
        {
            printf("jpg_decode_rgb error!\r\n");
            return -1;
        }

        dl_matrix3du_t org_img, gray_img;
        org_img.n = 1;
        org_img.c = 3;
        org_img.h = h;
        org_img.w = w;
        org_img.item = data;
        org_img.stride = org_img.w;
        gray_img.n = 1;
        gray_img.c = 1;
        gray_img.h = h;
        gray_img.w = w;
        gray_img.item = (uint8_t*)malloc(w*h);
        gray_img.stride = gray_img.w;
        rgb2gray(&org_img, &gray_img);
        free(org_img.item);
        
        vector<string> x = split(line, "/"); 
        //printf("%s \r\n", x[4].c_str()); 
        if(AddOneItem(&face_handle, x[4].c_str(), &gray_img) != 0)
        {
            printf("AddOneItem Error\r\n");
        }
    }
    fin.close();

    //搜库
    string test_file(argv[1]);
    string test_db_file_list  = pathvar + test_file;
    fin.open(test_db_file_list);
    while (getline(fin, line)) 
    {
        uint8_t* data;
        int w = 0;
        int h = 0;
        //string img_path = pathvar + line;
        string img_path = line;
        if( 0 == jpg_decode_rgb(img_path.c_str(), &data, &w, &h))
        {
            if ( w == 0 || h == 0 )
            {
                printf("jpg_decode_rgb w or h error!\r\n");
                return -1;
            }
        }
        else
        {
            printf("jpg_decode_rgb error!\r\n");
            return -1;
        }

        dl_matrix3du_t org_img, gray_img;
        org_img.n = 1;
        org_img.c = 3;
        org_img.h = h;
        org_img.w = w;
        org_img.item = data;
        org_img.stride = org_img.w;
        gray_img.n = 1;
        gray_img.c = 1;
        gray_img.h = h;
        gray_img.w = w;
        gray_img.item = (uint8_t*)malloc(w*h);
        gray_img.stride = gray_img.w;
        rgb2gray(&org_img, &gray_img);
        free(org_img.item);
        
        printf("Key:%s  RegScore: \r\n",img_path.c_str());
        if(SearchOneItem(&face_handle, &gray_img))
        {
            printf("Hit %s\r\n", line.c_str());
        }
        else
        {
            printf("Not Found %s\r\n", line.c_str());
        }
    }
    fin.close();

    return 0;
}
