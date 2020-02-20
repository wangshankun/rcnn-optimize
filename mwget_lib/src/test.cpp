

extern int mwget_entrance(char *url_orig, char *file_name = 0, char *file_dir = 0, int thread_num = 4);


int main()
{
    mwget_entrance("http://ait-public.oss-cn-hangzhou-zmf.aliyuncs.com/hci_team/wente.wwt/hrt.tar?OSSAccessKeyId=LTAIEWgqns5qyDNP&Expires=1582214274&Signature=hJmk21oHvc9kTeAP6eTHvW8G%2FiU%3D");
    return 0;
}
