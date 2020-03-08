#include <stdio.h>
#include <iostream>
#include <string.h>
#include <vector>

using namespace std;

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

int main(void) 
{
  
  string input_str = "rtsp://172.20.1.196:31162/11234567891320000005/20200305T170000Z/20200305T171000Z/PlayMode=filemode+FilePath=/mnt/BK/bm1nfs";
  vector<string> x = split(input_str, "/"); 
  
  for (int i = 0; i < x.size(); i++) 
  {
    fprintf(stderr, "%s \r\n", x[i].c_str());
  }

  return 0;
}
