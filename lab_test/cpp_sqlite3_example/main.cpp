#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sqlite3.h> 


static int sqlite_callback(void *data, int argc, char **argv, char **azColName){
   int i;
   fprintf(stderr, "%s: ", (const char*)data);
   for(i=0; i < argc; i++)
   {
      printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
   }
   printf("\n");
   return 0;
}

static int exec(sqlite3* db, const char* sql)
{
    int rc;
    char *zsqlite3_errmsg = 0;
    const char* callback_default_data = "sqlite callback function called";
    rc = sqlite3_exec(db, sql, sqlite_callback, (void*)callback_default_data, &zsqlite3_errmsg);
    if( rc != SQLITE_OK )
    {
      fprintf(stderr, "SQL error: %s\n", zsqlite3_errmsg);
      sqlite3_free(zsqlite3_errmsg);
    }
    return rc;
}

int db_create_employee_table(sqlite3 *pDb)
{
    const char*sql = "CREATE TABLE IF NOT EXISTS employee (  \
          uniqueid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, \
          id             INT, \
          name           varchar(255), \
          sex            varchar(255), \
          pic            varchar(255), \
          picName        varchar(255), \
          departmentId   INT, \
          status         INT, \
          createdTime    datetime, \
          updatedTime    datetime, \
          featureId      INT, \
          feature        BLOB)";

    return exec(pDb, sql);
}

int db_open(sqlite3 **ppDb, const std::string &path) 
{
    int rc = SQLITE_OK;
    if (path.empty())
    {
        rc = sqlite3_open(":memory", ppDb);
    }
    else
    {
        rc = sqlite3_open(path.c_str(), ppDb);
    }
    
    if (rc != SQLITE_OK)
    {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(*ppDb));
    }
    return rc;
}

int db_close(sqlite3 *pDb) 
{
    int rc = sqlite3_close(pDb);
    if (rc != SQLITE_OK)
    {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(pDb));
    }
    return rc;
}
/*
int db_encrypt(sqlite3 *pDb, const std::string &password) 
{
    int rc = SQLITE_OK;
    if (password.empty())
    {
        return -1;
    }
    else
    {
        rc = sqlite3_key(pDb, password.c_str(), (int)password.length());
    }
    if (rc != SQLITE_OK)
    {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(pDb));
    }
    return rc;
}
*/
std::string db_get_key(const std::string &path)
{
    struct stat buf;
    int result;
    result = stat(path.c_str(), &buf);
    if( result != 0 )
    {

    }
    else
    {
        printf("文件大小: %d", buf.st_size);
        printf("文件创建时间: %s", ctime(&buf.st_ctime));
        printf("访问日期: %s", ctime(&buf.st_atime));
        printf("最后修改日期: %s", ctime(&buf.st_mtime));
    }
    
    return ctime(&buf.st_ctime);
}

int main(int argc, char* argv[])
{
    std::string path = "./client.db";
    sqlite3 *client_db = nullptr;
    db_open(&client_db, path);
    std::string password = db_get_key(path);
    //db_encrypt(client_db, password);

    db_create_employee_table(client_db);


    sqlite3 *tmp_emp_db = nullptr;
    int rc = SQLITE_OK;
    rc = sqlite3_open("employees.db", &tmp_emp_db);
    if( rc )
    {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(tmp_emp_db));
    }

    char* sql = (char*)"select * from employee";
    exec(tmp_emp_db, sql);
    sql = (char*)"select * from sqlite_master where type=\"table\"";
    exec(tmp_emp_db, sql);

    //先把数据库做attach
    sql = (char*)"attach database \"./employees.db\" as tmp_emp_db;";
    exec(client_db, sql);

    sql = (char*)"attach database \"./client.db\" as client_db;";
    exec(client_db, sql);
    
    /*
    //client_db employee比 tmp_emp_db employee多了几列因此不能直接整体插入
    sql = (char*)"INSERT INTO client_db.employee SELECT * FROM tmp_emp_db.employee;";
    exec(client_db, sql);
    */
    
    //选择这种方式跨数据库attach插表， SELECT不能有括号
    sql = (char*)"INSERT INTO client_db.employee(id, \
                                                 name, \
                                                 sex, \
                                                 pic, \
                                                 picName, \
                                                 departmentId, \
                                                 status, \
                                                 createdTime, \
                                                 updatedTime) \
                  SELECT                        id, \
                                                name, \
                                                sex, \
                                                pic, \
                                                picName, \
                                                departmentId, \
                                                status, \
                                                createdTime, \
                                                updatedTime \
                  FROM tmp_emp_db.employee";

    exec(client_db, sql);
    
    sql = (char*)"select * from employee";
    exec(client_db, sql);
    
    sqlite3_close(client_db);
    
    return 0;
}
