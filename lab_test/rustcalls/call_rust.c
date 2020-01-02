//https://dev.to/dandyvica/how-to-call-rust-functions-from-c-on-linux-h37

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

// sample struct to illustrate passing a C-struct to Rust
struct CStruct {
    char c;
    unsigned long ul;
    char *s;
};

struct ByteBuffer {
    int64_t len;
    unsigned char *data; // note: nullable
};

typedef struct CompressInputImage  {
    char*  image_id;     
    char*  channel_id;   
    long   ts_ms;        
    int    compress_rate;
    int    image_format; 
    char*  buf;          
    unsigned long   buf_len;      
} CompressInputImage ;

// functions called in the Rust library
extern void rust_char(char c);
extern void rust_wchar(wchar_t c);
extern void rust_short(short i);
extern void rust_ushort(unsigned short i);
extern void rust_int(int i);
extern void rust_uint(unsigned int i);
extern void rust_long(long i);
extern void rust_ulong(unsigned long i);
extern void rust_string(char *s);
extern void rust_void(void *s);
extern void rust_int_array(const int *array, int length);
extern void rust_string_array(const char **array, int length);
extern void rust_cstruct(struct CStruct *c_struct);

extern void rust_buffer(struct ByteBuffer);
extern void rust_copimg(struct CompressInputImage*);
extern void rust_copimg_array(struct CompressInputImage*, unsigned int );

int main() {
    // pass char to Rust
    rust_char('A');
    rust_wchar(L'ζ');

    // pass short to Rust
    rust_short(-100);
    rust_ushort(100);

    // pass int to Rust
    rust_int(-10);
    rust_uint(10);

    // pass long to Rust
    rust_long(-1000);
    rust_ulong(1000);    

    // pass a NULL terminated string
    rust_string("hello world");

    // pass a void* pointer
    void *p = malloc(1000);
    rust_void(p);  

    // pass an array of ints
    int digits[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; 
    rust_int_array(digits, 10);

    // pass an array of c strings
    const char *words[] = { "This", "is", "an", "example"};
    rust_string_array(words, 4);   

    // pass a C struct
    struct CStruct c_struct;
    c_struct.c = 'A';
    c_struct.ul = 1000;
    c_struct.s = malloc(20);
    strcpy(c_struct.s, "0123456789");
    rust_cstruct(&c_struct);

    // don't forget to clean up ;-)
    free(p); 
    free(c_struct.s);

    struct ByteBuffer testbuf;
    testbuf.data =  malloc(20);
    testbuf.len = 20;
    memset(testbuf.data,'c',20);
    rust_buffer(testbuf);

    CompressInputImage tt;
    tt.image_id = "aadf";
    tt.channel_id = "1233_sdsd";
    tt.ts_ms      = 6666666;
    tt.compress_rate = 17;
    tt.image_format =0;
    tt.buf          =  malloc(20);
    memset(tt.buf,'c',20);
    tt.buf_len      = 20;
    rust_copimg(&tt);

    CompressInputImage tts[3];
    tts[0].image_id = "aadf1";
    tts[0].channel_id = "111111233_sdsd";
    tts[0].ts_ms      = 6666666;
    tts[0].compress_rate = 17;
    tts[0].image_format =0;
    tts[0].buf          =  malloc(20);
    memset(tts[0].buf,'a',20);
    tts[0].buf_len      = 20;

    tts[1].image_id = "aadf2";
    tts[1].channel_id = "222221233_sdsd";
    tts[1].ts_ms      = 6666666;
    tts[1].compress_rate = 17;
    tts[1].image_format =0;
    tts[1].buf          =  malloc(21);
    memset(tts[1].buf,'b',21);
    tts[1].buf_len      = 21;
    
    tts[2].image_id = "aadf3";
    tts[2].channel_id = "33331233_sdsd";
    tts[2].ts_ms      = 6666666;
    tts[2].compress_rate = 17;
    tts[2].image_format =0;
    tts[2].buf          =  malloc(22);
    memset(tts[2].buf,'c',22);
    tts[2].buf_len      = 22;

    rust_copimg_array(tts, 3);

    return 0;
}
