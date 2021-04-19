#include "http_client.h"
int main()
{
    HttpClient client("http://47.96.252.26:8029/mapi/", "12345", "54321", "./");
    
    client.init();//初始化

    thread client_exe_heart_beat(client_thread_heart_beat, &client);
    thread client_exe_sycn_data(client_thread_sync_data, &client);
    client_exe_heart_beat.join();//心跳线程
    client_exe_sycn_data.join();//数据同步线程
}
