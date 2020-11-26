#ifndef HTTP_CLIENT_H_
#define HTTP_CLIENT_H_

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <json/json.h>
#include <curl/curl.h>
#include <curl/easy.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <openssl/md5.h>

using namespace std;

#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef size_t (*CURL_CALLBACK_FUNC)(void *ptr, size_t size, size_t nmemb, void *this_ptr);

enum RUNNING_STATUS
{
  HEART_BEAT = 1,
  SYNC_DATA, 
  INSERT_ID_HISTORY,
  EXIT,
};

typedef struct {
    RUNNING_STATUS status; //状态机
    uint64_t server_currentVersion;//服务器端目前版本(时间戳)
    uint64_t downloading_syncTime;//下载数据时候同步的时间戳
    int32_t  get_data_batchSize;//一次请求同步数据个数(期望值)
    uint64_t lastSyncTime;//设备端上次更新的时间
    int32_t  server_ret_code;//post请求后返回的code码
    string   message;//post请求后返回的message信息，是对code的码的解释
    int32_t  downloading_ret;//一次下载是否成功
    string   dataFile;//下载的文件链接的路径
    uint64_t downloading_retry_times;//下载不成功次数

    int32_t identified_employeeId;//识别到的员工号
    int32_t identified_result;//识别到状态
} running_status_arg_t;

class HttpClient {
    public:
        HttpClient(string host_url, string appKey, string appSecret, string database_path) 
        : host_url_(host_url), appKey_(appKey), appSecret_(appSecret), database_path_(database_path)
        {}
        int32_t init();
        void    run();

    friend void client_thread_heart_beat(HttpClient *client);
    friend void client_thread_sync_data(HttpClient *client);

    private:
        mutex running_arg_mtx_;
        string database_path_;
        string appKey_;
        string appSecret_;
        string host_url_;

        string fingerPrint_;
        string status_;
        string opSystem_;
        string ip_;
        running_status_arg_t running_arg_;
        
        static size_t post_callback(void *ptr, size_t size, size_t nmemb, HttpClient *this_ptr);
        static size_t downloading_callback(void *ptr, size_t size, size_t nmemb, HttpClient *this_ptr);
        int32_t curl_form_urlencoded_post(string body_str, string url_str);
        int32_t curl_form_download_post(string body_str, string url_str);
        
        int32_t post_heart_beat();
        int32_t post_sync_data();
        int32_t post_download_file();
        int32_t post_identified_result();
};

void client_thread_heart_beat(HttpClient *client);
void client_thread_sync_data(HttpClient *client);

static uint64_t get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000+tv.tv_usec/1000;
}

static string common_md5_secret32(const string& src)
{
    #define MD5_SECRET_LEN_16     (16)
    #define MD5_BYTE_STRING_LEN   (4)

    MD5_CTX ctx;
    string md5String;
    unsigned char md[MD5_SECRET_LEN_16] = { 0 };
    char tmp[MD5_BYTE_STRING_LEN] = { 0 };
 
    MD5_Init( &ctx );
    MD5_Update( &ctx, src.c_str(), src.size() );
    MD5_Final( md, &ctx );
 
    for( int32_t i = 0; i < 16; ++i )
    {
        memset( tmp, 0x00, sizeof( tmp ) );
        snprintf( tmp,sizeof(tmp) , "%02x", md[i] );
        md5String += tmp;
    }
    return md5String;
}

static void print_hex(const void* buf , size_t size)
{
    unsigned char* str = (unsigned char*)buf;
    char line[512] = {0};
    const size_t lineLength = 16; // 8或者32
    char text[24] = {0};
    char* pc;
    int32_t textLength = lineLength;
    size_t ix = 0 ;
    size_t jx = 0 ;

    for (ix = 0 ; ix < size ; ix += lineLength) {
        sprintf(line, "%.8xh: ", ix);
// 打印16进制
        for (jx = 0 ; jx != lineLength ; jx++) {
            if (ix + jx >= size) {
                sprintf(line + (11 + jx * 3), "   "); // 处理最后一行空白
                if (ix + jx == size)
                    textLength = jx;  // 处理最后一行文本截断
            } else
                sprintf(line + (11 + jx * 3), "%.2X ", * (str + ix + jx));
        }
// 打印字符串
        {
            memcpy(text, str + ix, lineLength);
            pc = text;
            while (pc != text + lineLength) {
                if ((unsigned char)*pc < 0x20) // 空格之前为控制码
                    *pc = '.';                 // 控制码转成'.'显示
                pc++;
            }
            text[textLength] = '\0';
            sprintf(line + (11 + lineLength * 3), "; %s", text);
        }

        printf("%s\n", line);
    }
}

static vector<string> split(const string& str, const string& delim) 
{
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

//2的幂次秒等待，最多512秒
static uint64_t exe_retry_wait_time(uint64_t retry_times)
{
    if(retry_times < 10)
    {
        std::this_thread::sleep_for(std::chrono::seconds(2 << retry_times));
    }
    else
    {
        std::this_thread::sleep_for(std::chrono::seconds(2 << 9));
    }
}

#endif
