#include <ARC4.h>
#include <iostream>
#include <string>
#include <malloc.h>
#include <string.h>

using std::string;
using std::cout;

int main ()
{
/*
    string someKey  = "I_AM_A_KEY";
    string toEncrypt = "I am a plain text string";
    ARC4 rc4;
    rc4.setKey((unsigned char*) someKey.c_str(), someKey.length());
    char * enc=  (char *) malloc(toEncrypt.size()+1);
    char * dec=  (char *) malloc(toEncrypt.size()+1);

    rc4.encrypt((char *)  toEncrypt.c_str(), enc, toEncrypt.length());
    cout<<" Encrypted result \n";
    cout << enc << "\n";

    rc4.setKey((unsigned char*) someKey.c_str(),someKey.length());
    rc4.encrypt(enc , dec, toEncrypt.length());

    cout<<" Decrypted result \n";
    cout << dec << "\n";
    if(strcmp(dec, toEncrypt.c_str()) ==0){
        cout<<"Enctyption successfull!\n";
    }
    else{
        cout<<"Something is wrong on the encriptyon!\n";
    }
*/
    ARC4 rc4;
    char* key = (char*){"jaoijdfioasdfj"};
    rc4.setKey((unsigned char*)key, strlen(key));
    char * src=  (char *) malloc(128000);
    char * enc=  (char *) malloc(128000);
    char * dec=  (char *) malloc(128000);

    rc4.encrypt(src, enc, 128000);

    rc4.setKey((unsigned char*)key, strlen(key));
    rc4.encrypt(enc, dec, 128000);

    if(strcmp(dec, src) ==0){
        cout<<"Enctyption successfull!\n";
    }
    else{
        cout<<"Something is wrong on the encriptyon!\n";
    }
}
