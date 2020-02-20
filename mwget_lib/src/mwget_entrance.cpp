/*  MWget - A Multi download for all POSIX systems.
 *  Homepage: http://mwget.sf.net
 *  Copyright (C) 2005- rgwan,xiaosuo
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

#include <iostream>
#include <getopt.h>
#include <signal.h>

#include "mwget.h"
#include "initi18n.h"
using namespace std;

int mwget_entrance(char *url_orig, char *file_name = NULL, char *file_dir = NULL, int thread_num = 4)
{
    int ret;
    URL url;
    Downloader downloader;
    Task task;
    Proxy proxy;
    char *ptr = NULL;
    seti18npackage();//设置国际化
    signal(SIGPIPE, SIG_IGN);
#ifdef HAVE_SSL
    SSL_load_error_strings();
    SSLeay_add_ssl_algorithms();
#endif

    global_debug = false;
    
    task.tryCount = 3;
    if (file_dir!= NULL)  task.set_local_dir(file_dir);
    if (file_name!= NULL) task.set_local_file(file_name);
    task.retryInterval = 1;
    task.threadNum = thread_num;//默认下载进程数
    task.timeout = 3000;

    ptr = NULL;//设置代理

    if(ptr == NULL){
        ptr = StrDup(getenv("proxy"));
    }
    if(ptr){
        if(url.set_url(ptr) < 0){
            delete[] ptr;
            cerr<<_("!!!Please check your http_proxy setting!")<<endl;
            return -1;
        }
        delete[] ptr;
        if(url.get_protocol() != HTTP){
            cerr<<_("!!!The proxy type is not supported")<<endl;
            return -1;
        }
        proxy.set_type(HTTP_PROXY);
        proxy.set_host(url.get_host());
        proxy.set_port(url.get_port());
        proxy.set_user(url.get_user());
        proxy.set_password(url.get_password());
        task.proxy = proxy;
    }

    if(url.set_url(url_orig) < 0){
        cerr<<_("!!!set_url failed")<<endl;
        return -1;
    }
    task.url = url;
    downloader.task = task;
    downloader.run();

    return 0;
};
