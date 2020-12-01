#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h>

static void print_hex(const void* buf , size_t size)
{
    unsigned char* str = (unsigned char*)buf;
    char line[512] = {0};
    const size_t lineLength = 16; // 8或者32
    char text[24] = {0};
    char* pc;
    int textLength = lineLength;
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

static int sqlite_callback(void *data, int argc, char **argv, char **azColName){
   int i;
   fprintf(stderr, "%s: ", (const char*)data);
   for(i=0; i<argc; i++){
      printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
   }
   printf("\n");
   return 0;
}

static int exec(sqlite3* db, const char* sql)
{
    int rc;
    char *zErrMsg = 0;
    const char* callback_default_data = "sqlite callback function called";
    rc = sqlite3_exec(db, sql, sqlite_callback, (void*)callback_default_data, &zErrMsg);
    if( rc != SQLITE_OK ){
      fprintf(stderr, "SQL error: %s\n", zErrMsg);
      sqlite3_free(zErrMsg);
    }else{
      fprintf(stdout, "Operation done successfully\n");
    }
    return rc;
}

int main(int argc, char* argv[])
{
   sqlite3 *db;
   char *sql;
   int rc;

   /* Open database */
   rc = sqlite3_open("employees.db", &db);
   if( rc ){
      fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
      exit(0);
   }else{
      fprintf(stderr, "Opened database successfully\n");
   }

   sql = "select * from sqlite_master where type=\"table\"";
   exec(db, sql);

   sql = "select * from employee";
   exec(db, sql);

   sql = "ALTER TABLE employee ADD COLUMN feature Blob";//增加一列存储二进制特征值
   exec(db, sql);

   sql = "PRAGMA TABLE_INFO (employee)";
   exec(db, sql);
   
   sql = "select * from employee where pic like '%wangwu.jpg%' ";
   exec(db, sql);
   
   //内存数据写入blob
   sqlite3_stmt*  stmt = NULL;
   unsigned char buf[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
   sqlite3_prepare(db, "UPDATE employee set feature = ? where pic like '%wangwu.jpg%' ", -1, &stmt, NULL);
   sqlite3_bind_blob(stmt, 1, buf, 16, NULL);
   sqlite3_step(stmt);

   sql = "select * from employee where pic like '%wangwu.jpg%' ";
   exec(db, sql);


    //从数据库中读取blob
    char *data=NULL;
    int len = 0;
    sqlite3_prepare(db, "select feature from employee where pic like '%wangwu.jpg%' ", -1, &stmt, NULL);
    sqlite3_step(stmt);
    data=(unsigned char *)sqlite3_column_blob(stmt, 0); //得到纪录中的BLOB字段
    len=sqlite3_column_bytes(stmt, 0);//得到字段中数据的长度
    print_hex(data, len);//memmove(buffer,data,len);
    
    sqlite3_close(db);
    return 0;
}
