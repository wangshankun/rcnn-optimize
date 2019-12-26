#include<stdio.h>
#include<stdlib.h>
#include<string.h>

void fenge(char*dst,char* cat)
{
     static int ids_byte_len = 1;
     ids_byte_len = ids_byte_len + strlen(cat) + 1;
     printf("ids_byte_len:%d \r\n",ids_byte_len);
     dst = realloc(dst, ids_byte_len);
     dst = strcat(dst, cat);
     dst = strcat(dst, ";");
}
void main()
{
 char* s = malloc(1);
 char* a = "123456";
 char* b = "12345";
 char* c = "1234";
 
 fenge(s,a);
 printf("%s\r\n",s);
 fenge(s,b);
 printf("%s\r\n",s);
 fenge(s,c);
 printf("%s\r\n",s);
}
