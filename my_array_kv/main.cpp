#include "SimpleHash.h"

#define db_file "test.db"
#define FACE_DNA_KEY "face_dna_key"
const int TEST_TABLE_SIZE = 10000;

int main()
{
	//Perfectly horrendous way of seeding a PRNG, but we'll do it here for brevity
	auto generator = mt19937(12345); //Combination of my luggage
	//Generate values within the specified closed intervals

	SimpleHash map(TEST_TABLE_SIZE * (1.0/0.7));//1.42倍的hash容量
	cout<< "Created table of size "<< map.size()<<endl;

	cout<<"Generating test data..."<<endl;
//=========================测试insert接口==================================================
	for(int i = 0; i < TEST_TABLE_SIZE; i++)
	{
		array<char, 64> key = {0};
		sprintf(key.data(), "%s_%d", FACE_DNA_KEY, i);
                string key_str(key.data()); 
		
		array<float, 64> value;
		for(int i = 0; i < 64; ++i)
		{
			uniform_real_distribution<float> dist(0.0, 9.9);
			value[i] = dist(generator);
		}

		map.insert(key_str, value); //Low chance of collisions, so we get quite close to the desired size
	}

//=================测试文件读写==============================
	{
		const auto start = chrono::steady_clock::now();
		map.tofile(db_file);
		const auto end = chrono::steady_clock::now();
		cout<<"Save time = "<< chrono::duration<double, milli> (end-start).count() << " ms" << endl;
	}
	{
		const auto start = chrono::steady_clock::now();
		map.loadfile(db_file);
		const auto end = chrono::steady_clock::now();
		cout<<"Load time = "<< chrono::duration<double, milli> (end-start).count() << " ms" << endl;
	}
//=================测试文件读写正确性==============================
	string test_key = "face_dna_key_777";
	array<float, 64> test_value = {0};
	if(map.get(test_key, test_value) == 0)
	{
		printf("face_dna_key_777 's value:\r\n");
		for(int j = 0; j < 64; j++)
		{
			if(j%8 == 0 && j != 0) printf("\r\n");
			printf("%.8f ", test_value.at(j));
		}
		printf("\r\n");
	}
	else
	{
		std::cout<<"Get Failed, Not Found"<<std::endl;
	}
	
    {
		SimpleHash newmap, n_newmap;//测试空map从文件加载
		const auto start = chrono::steady_clock::now();
		n_newmap.loadfile(db_file);
		const auto end = chrono::steady_clock::now();
		cout<<"New Load time = "<< chrono::duration<double, milli> (end-start).count() << " ms" << endl;
                newmap = n_newmap;//测试拷贝构造

		string test_key = {"face_dna_key_777"};
		array<float, 64> test_value = {0};
		if(newmap.get(test_key, test_value) == 0)
		{
			printf("face_dna_key_777 's value:\r\n");
			for(int j = 0; j < 64; j++)
			{
				if(j%8 == 0 && j != 0) printf("\r\n");
				printf("%.8f ", test_value.at(j));
			}
			printf("\r\n");
		}
		else
		{
			std::cout<<"Get Failed, Not Found"<<std::endl;
		}
    }
//=========================测试remove接口==================================================
	if(map.remove(test_key) == 0)//先删除
	{
	}
	else
	{
		cout<<"Remove Failed, Not Found"<< endl;
	}

	if(map.get(test_key, test_value) == 0)//再查找
	{

	}
	else
	{
		cout<<"Get Failed, Not Found"<< endl;
	}
}
