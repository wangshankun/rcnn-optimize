#include "http_client.h"

size_t HttpClient::downloading_callback(void* pBuffer, size_t nSize, size_t nMemByte, HttpClient *this_ptr)
{
    string file_path = this_ptr->running_arg_.dataFile;
    vector<string> file_path_spl = split(file_path, "/"); //斜线分割符，取最后一个是文件名
    string file_name = file_path_spl[file_path_spl.size() - 1];
    string save_file_path = this_ptr->database_path_ + file_name;
    
    FILE* fp = fopen(save_file_path.c_str(), "ab+");
    if(fp == NULL)  
    {
        this_ptr->running_arg_mtx_.lock();
        this_ptr->running_arg_.downloading_ret = -1;
        this_ptr->running_arg_mtx_.unlock();
        fprintf(stderr, "%s  %d  open file %s failed! \r\n",__FUNCTION__, __LINE__, save_file_path.c_str());
        return -1;//EOF
    }
    else
    {
        size_t nWrite = fwrite(pBuffer, nSize, nMemByte, fp);
        if(nWrite < 0)
        {
            fclose(fp);
            this_ptr->running_arg_mtx_.lock();
            this_ptr->running_arg_.downloading_ret = -1;
            this_ptr->running_arg_mtx_.unlock();
            fprintf(stderr, "%s  %d  write file %s failed! \r\n",__FUNCTION__, __LINE__, save_file_path.c_str());
            return -1;
        }
        else
        {
            fclose(fp);
            this_ptr->running_arg_mtx_.lock();
            this_ptr->running_arg_.downloading_ret = 0;
            this_ptr->running_arg_mtx_.unlock();
            return nWrite;
        }
    }
}

int32_t HttpClient::curl_form_download_post(string body_str, string url_str)
{
    CURL_CALLBACK_FUNC callback = (CURL_CALLBACK_FUNC)(&HttpClient::downloading_callback);

    string tmstp_str = to_string(get_current_time());
    string form = appKey_ + appSecret_ + body_str + tmstp_str;
    string sign = common_md5_secret32(form);
    
    CURL *pCurlHandle = curl_easy_init();
    curl_easy_setopt(pCurlHandle, CURLOPT_CUSTOMREQUEST, "POST");
    curl_easy_setopt(pCurlHandle, CURLOPT_URL, url_str.c_str());
    curl_easy_setopt(pCurlHandle, CURLOPT_LOW_SPEED_TIME, 10);//设置超时时间
        
    struct curl_slist *pCurlList = NULL;
    pCurlList = curl_slist_append(pCurlList, "Content-Type: application/x-www-form-urlencoded");//指定文本url编码
    curl_easy_setopt(pCurlHandle, CURLOPT_HTTPHEADER, pCurlList);

    //管控台只将body做了escape解码, appkey等其他内容明文发送;
    char*  psz_encode_body  = curl_easy_escape(pCurlHandle, body_str.c_str(), body_str.length());
    string str_encode_body = psz_encode_body;
    curl_free(psz_encode_body);//释放申请的内存

    string strPostData = "body=" + str_encode_body + 
                         "&appKey=" + appKey_ + 
                         "&appSecret=" + appSecret_ + 
                         "&timestamp=" + tmstp_str +
                         "&sign=" + sign;

    curl_easy_setopt(pCurlHandle, CURLOPT_POSTFIELDS, strPostData.c_str());
    curl_easy_setopt(pCurlHandle, CURLOPT_WRITEFUNCTION,  callback);
    curl_easy_setopt(pCurlHandle, CURLOPT_WRITEDATA, (void *)this);

    CURLcode nRet = curl_easy_perform(pCurlHandle);
    if (0 != nRet)
    {
        running_arg_mtx_.lock();
        running_arg_.downloading_ret = -1;
        running_arg_mtx_.unlock();
        fprintf(stderr, "%s  %d  libcurl curl_easy_perform  post message failed!  %s\r\n",__FUNCTION__, __LINE__ , curl_easy_strerror(nRet));
        return -1;
    }
    if(0 != running_arg_.downloading_ret)//发送成功后，立即通过callback函数将服务器端执行结果返回(例子是单进程阻塞同步方式)
    {
       fprintf(stderr, "%s  %d  url:%s download %s failed! \r\n",__FUNCTION__, __LINE__, url_str.c_str(), running_arg_.dataFile);
       return -1;
    }
    curl_slist_free_all(pCurlList);
    curl_easy_cleanup(pCurlHandle);

     return 0;
}

size_t HttpClient::post_callback(void *ptr, size_t size, size_t nmemb, HttpClient *this_ptr)
{ 
    string ret_info((const char*) ptr, (size_t) size * nmemb);
    print_hex(ptr, size * nmemb);

    bool res;
    JSONCPP_STRING errs;
    Json::Value root, data, department;
    Json::CharReaderBuilder readerBuilder;

    std::unique_ptr<Json::CharReader> const jsonReader(readerBuilder.newCharReader());
    res = jsonReader->parse(ret_info.c_str(), ret_info.c_str() + ret_info.length(), &root, &errs);
    if (!res || !errs.empty()) {
        std::cout << "parseJson err. " << errs << std::endl;
    }

    this_ptr->running_arg_mtx_.lock();
    this_ptr->running_arg_.server_ret_code    = root["code"].asInt();
    this_ptr->running_arg_.message            = root["message"].asString();
    //running_arg_.success = root["success"].asBool();

    if(root.isMember("data"))
    {
        if(root["data"].isNull())
        {
           //pass;
        }
        else
        {
            data = root["data"];
            if(data.isMember("department"))
            {
                department = data["department"];
                
                if(department.isMember("currentVersion"))
                {
                    this_ptr->running_arg_.server_currentVersion = department["currentVersion"].asUInt64();
                }
                else if(department.isMember("dataFile"))
                {
                    this_ptr->running_arg_.dataFile = department["dataFile"].asString();
                    this_ptr->running_arg_.downloading_syncTime = department["syncTime"].asUInt64();
                }
            }
        }
    }
    else
    {
        //pass;
    }
    this_ptr->running_arg_mtx_.unlock();
    
    return size * nmemb;
}

int32_t HttpClient::curl_form_urlencoded_post(string body_str, string url_str)
{
    CURL_CALLBACK_FUNC callback = (CURL_CALLBACK_FUNC)(&HttpClient::post_callback);
  
    string tmstp_str = to_string(get_current_time());
    string form = appKey_ + appSecret_ + body_str + tmstp_str;
    string sign = common_md5_secret32(form);
 
    CURL *pCurlHandle = curl_easy_init();
    curl_easy_setopt(pCurlHandle, CURLOPT_CUSTOMREQUEST, "POST");
 
    curl_easy_setopt(pCurlHandle, CURLOPT_URL, url_str.c_str());
 
    curl_easy_setopt(pCurlHandle, CURLOPT_LOW_SPEED_TIME, 10);//设置超时时间
        
    struct curl_slist *pCurlList = NULL;
    pCurlList = curl_slist_append(pCurlList, "Content-Type: application/x-www-form-urlencoded");//指定文本url编码
    curl_easy_setopt(pCurlHandle, CURLOPT_HTTPHEADER, pCurlList);

    //管控台只将body做了escape解码, appkey等其他内容明文发送;
    char*  psz_encode_body  = curl_easy_escape(pCurlHandle, body_str.c_str(), body_str.length());
    string str_encode_body = psz_encode_body;
    curl_free(psz_encode_body);//释放申请的内存
    string strPostData = "body=" + str_encode_body + 
                         "&appKey=" + appKey_ + 
                         "&appSecret=" + appSecret_ + 
                         "&timestamp=" + tmstp_str +
                         "&sign=" + sign;

    curl_easy_setopt(pCurlHandle, CURLOPT_POSTFIELDS, strPostData.c_str());
    curl_easy_setopt(pCurlHandle, CURLOPT_WRITEFUNCTION,  callback);
    curl_easy_setopt(pCurlHandle, CURLOPT_WRITEDATA, (void *)this);

    try
    {
        CURLcode nRet = curl_easy_perform(pCurlHandle);
        if (0 != nRet)
        {
            fprintf(stderr, "%s  %d  libcurl curl_easy_perform  post message failed!  %s\r\n",__FUNCTION__, __LINE__ , curl_easy_strerror(nRet));
            return -1;
        }
    }
    catch (const char* msg)
    {
        fprintf(stderr, "%s  %d  libcurl curl_easy_perform  post crash  %s\r\n",__FUNCTION__, __LINE__ , msg);
    }

    if(running_arg_.server_ret_code != 1000)//发送成功后，立即通过callback函数将服务器端执行结果返回(例子是单进程阻塞同步方式)
    {
       fprintf(stderr, "%s  %d  url:%s request post failed! \r\n",__FUNCTION__, __LINE__, url_str.c_str());
       return -1;
    }

    curl_slist_free_all(pCurlList);
 
    curl_easy_cleanup(pCurlHandle);

    return 0;
}

int32_t HttpClient::post_identified_result()
{
    Json::Value body_json;
    Json::Value identifiedResult_json;
    identifiedResult_json["fingerPrint"] =  fingerPrint_;
    identifiedResult_json["employeeId"]  =  running_arg_.identified_employeeId;
    identifiedResult_json["result"]      =  running_arg_.identified_result;
    
    body_json["identifiedResult"] = identifiedResult_json;
    
    string body_str = body_json.toStyledString();
    string dst_url = host_url_ + "insertIdentifiedHistory";
    return curl_form_urlencoded_post(body_str, dst_url);
}

int32_t HttpClient::post_heart_beat()
{
    Json::Value body_json;
    Json::Value equipment_json;
    Json::Value department_json;

    equipment_json["fingerPrint"] = fingerPrint_;
    equipment_json["status"] = status_;
    equipment_json["opSystem"] = opSystem_;
    equipment_json["ip"] = ip_;
    department_json["lastSyncTime"] = running_arg_.lastSyncTime;

    body_json["equipment"]  = equipment_json;
    body_json["department"] = department_json;
    string body_str = body_json.toStyledString();

    string dst_url = host_url_ + "heartBeat";
    return curl_form_urlencoded_post(body_str, dst_url);
}

int32_t HttpClient::post_sync_data()
{
    Json::Value body_json;
    Json::Value department_json;

    department_json["lastSyncTime"] = running_arg_.lastSyncTime;
    department_json["batchSize"] = running_arg_.get_data_batchSize;
    
    body_json["department"] = department_json;
    string body_str = body_json.toStyledString();
    string dst_url = host_url_ + "getSyncDataFile";
 
    return curl_form_urlencoded_post(body_str, dst_url);
}

int32_t HttpClient::post_download_file()
{
    Json::Value body_json;
    body_json["path"] = running_arg_.dataFile;
    string body_str = body_json.toStyledString();
    string dst_url = host_url_ + "download";
    return curl_form_download_post(body_str, dst_url);
}

int32_t HttpClient::init()
{
    fingerPrint_ = "xxxx23322xxxx"; //每次开机从数据中获得
    status_ = "1";
    opSystem_ = "Linux";
    ip_ = "11.1.1.1";//每次开机从系统中获得

    running_arg_.lastSyncTime = 0; //每次开机从本地数据库加载进去
    running_arg_.get_data_batchSize = 100;//根据数据库处理能力变化
    running_arg_.downloading_retry_times = 0;//下载失败次数为0
}

void client_thread_heart_beat(HttpClient *client)
{
    while (client->running_arg_.status != EXIT)
    {
        client->post_heart_beat();
        std::this_thread::sleep_for(std::chrono::milliseconds(10000));//定时发送心跳
    }
}

void client_thread_sync_data(HttpClient *client)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));//等待心跳线程先执行

    while (client->running_arg_.status != EXIT)
    {
        if(client->running_arg_.lastSyncTime < client->running_arg_.server_currentVersion)
        {
            if(0 != client->running_arg_.downloading_retry_times)
            {
                if(0 == client->post_download_file())
                {
                    client->running_arg_mtx_.lock();
                    //执行下载程序，下载成功结束后修改lastSyncTime为此次更新的时间
                    client->running_arg_.lastSyncTime = client->running_arg_.downloading_syncTime;
                    client->running_arg_.downloading_ret = 0;//设置下载成功标志
                    client->running_arg_.downloading_retry_times = 0;//退出重试状态
                    client->running_arg_mtx_.unlock();
                }
                else
                {
                    if(client->running_arg_.status == EXIT)
                    {
                        break;//强制退出线程
                    }
                    exe_retry_wait_time(client->running_arg_.downloading_retry_times);
                    client->running_arg_mtx_.lock();
                    client->running_arg_.downloading_retry_times++;
                    client->running_arg_mtx_.unlock();
                }
            }
            else if (client->post_sync_data() == 0 && client->running_arg_.dataFile != "")//有新数据得到
            {
                   if (0 == client->post_download_file())
                   {
                        client->running_arg_mtx_.lock();
                        //执行下载程序，下载成功结束后修改lastSyncTime为此次更新的时间
                        client->running_arg_.lastSyncTime = client->running_arg_.downloading_syncTime;
                        client->running_arg_.downloading_ret = 0;//设置下载成功标志
                        client->running_arg_mtx_.unlock();
                   }
                   else
                   {
                       client->running_arg_mtx_.lock();
                       //陷入SYNC_DATA的retry状态直到下载成功为止;
                       client->running_arg_.downloading_retry_times++;
                       client->running_arg_mtx_.unlock();
                   }
            }
        }
    }
}
