#include <iostream>
#include <sstream>
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
#include <boost/iostreams/filter/stdio.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/interprocess/streams/bufferstream.hpp>

#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>

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



bool Base64Encode(const string& input, string* output) {
  typedef boost::archive::iterators::base64_from_binary<boost::archive::iterators::transform_width<string::const_iterator, 6, 8> > Base64EncodeIterator;
  stringstream result;
  copy(Base64EncodeIterator(input.begin()) , Base64EncodeIterator(input.end()), ostream_iterator<char>(result));
  size_t equal_count = (3 - input.length() % 3) % 3;
  for (size_t i = 0; i < equal_count; i++) {
    result.put('=');
  }
  *output = result.str();
  return output->empty() == false;
}
 
bool Base64Decode(const string& input, string* output) {
  typedef boost::archive::iterators::transform_width<boost::archive::iterators::binary_from_base64<string::const_iterator>, 8, 6> Base64DecodeIterator;
  stringstream result;
  try {
    copy(Base64DecodeIterator(input.begin()) , Base64DecodeIterator(input.end()), ostream_iterator<char>(result));
  } catch(...) {
    return false;
  }
  *output = result.str();
  return output->empty() == false;
}

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

    std::vector<std::vector<uint8_t>> ser_jpgs;
    for (auto &d: files) 
    {
        string full_file_name = dir + d;
        int file_size = get_file_size(full_file_name.c_str());
        
        uint8_t* tmp = (uint8_t*)malloc(file_size);
        readfile(full_file_name.c_str(), tmp,  file_size);
        std::vector<uint8_t> jpg_file(&tmp[0], &tmp[file_size]);
        ser_jpgs.push_back(jpg_file);
        //printf("%s file_size:%d\r\n",full_file_name.c_str(), file_size);
    }

    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << ser_jpgs;
    //printf("ss.size:%d \r\n", ss.str().size());


    //#define tmp_buf_len (1024*1024*3L)
    //char* tmp = (char*)malloc(tmp_buf_len);
    //// write the serializable structure
    //boost::interprocess::obufferstream obs(static_cast<char*>(tmp), tmp_buf_len);
    //boost::archive::binary_oarchive oa(dynamic_cast<ostream&>(obs));
    //oa << ser_jpgs;

    //std::string serial_str;
    //boost::iostreams::back_insert_device<std::string> inserter(serial_str);
    //boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > oss(inserter);
    //boost::archive::binary_oarchive oa(oss);
    //oa << ser_jpgs;
    //oss.flush();
    //printf("serial_str.size:%d \r\n",serial_str.size());
    
    //序列化
    //std::ofstream ofs("/tmp/copy.ser");
    //boost::archive::binary_oarchive oa(ofs);
    //oa & ser_jpgs;

    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("ser elapsed time:%f\r\n",elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);
    std::string base64_str, output_str;
    Base64Encode(ss.str(), &base64_str);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Base64Encode elapsed time:%f\r\n",elapsed);
    

    Base64Decode(base64_str, &output_str);
    
    
    clock_gettime(CLOCK_MONOTONIC, &start);
     //反序列化
    std::vector<std::vector<uint8_t>> rser_jpgs;
    //std::ifstream ifs("/tmp/copy.ser");
    //boost::archive::binary_iarchive ia(ifs);
    //ia & rser_jpgs;

    //boost::iostreams::basic_array_source<char> device(serial_str.data(), serial_str.size());
    //boost::iostreams::stream<boost::iostreams::basic_array_source<char> > iss(device);
    //boost::archive::binary_iarchive ia(iss);
    //ia >> rser_jpgs;

    //boost::interprocess::ibufferstream ibs(static_cast<char*>(tmp), tmp_buf_len);
    //boost::archive::binary_iarchive ia(dynamic_cast<istream&>(ibs));
    //ia >> rser_jpgs;

    std::stringstream iss;
    iss << output_str;
    boost::archive::binary_iarchive ia(iss);
    ia >> rser_jpgs;
    printf("ss.size:%d %s \r\n", iss.str().size(), iss.str().c_str());
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("rser elapsed time:%f\r\n",elapsed);
    
   //resave last pic to check
   for (auto &d: rser_jpgs) 
   {
      savefile("vect_b.jpg", &d[0], d.size());
   }

  return 0;
}
