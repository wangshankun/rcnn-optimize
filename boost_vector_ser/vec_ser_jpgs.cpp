#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>

#include <sys/stat.h>
// include input and output archivers
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

// include this header to serialize vectors
#include <boost/serialization/vector.hpp>

using namespace std;

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


int get_dir_list(string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        if(dirp->d_type = 8 && dirp->d_reclen == 32) files.push_back(string(dirp->d_name));
        //printf(" d_name :%s  d_reclen:%d  d_type:%d\r\n", dirp->d_name, dirp->d_reclen, dirp->d_type);
    }
    closedir(dp);
    return 0;
}

int get_file_size(const char* file_name)
{
    struct stat statbuf;

    if (stat(file_name, &statbuf) == -1) 
    {
      return -1;
    }

    return statbuf.st_size;
}

int main()
{
    string dir = string("./test/");
    vector<string> files = vector<string>();
    get_dir_list(dir, files);

    std::vector<std::vector<uint8_t>> jpgs;
    for (auto &d: files) 
    {
        string full_file_name = dir + d;
        int file_size = get_file_size(full_file_name.c_str());
        
        uint8_t* tmp = (uint8_t*)malloc(file_size);
        readfile(full_file_name.c_str(), tmp,  file_size);
        std::vector<uint8_t> jpg_file(&tmp[0], &tmp[file_size]);
        jpgs.push_back(jpg_file);
        //printf("%s file_size:%d\r\n",full_file_name.c_str(), file_size);
    }

    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    //序列化
    std::ofstream ofs("/tmp/copy.ser");
    boost::archive::binary_oarchive oa(ofs);
    oa & jpgs;

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f\r\n",elapsed);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
     //序列化
    std::vector<std::vector<uint8_t>> rjpgs;
    std::ifstream ifs("/tmp/copy.ser");
    boost::archive::binary_iarchive ia(ifs);
    ia & rjpgs;
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f\r\n",elapsed);
    
   //resave last pic to check
   for (auto &d: rjpgs) 
   {
      savefile("vect_b.jpg", &d[0], d.size());
   }

  return 0;
}
