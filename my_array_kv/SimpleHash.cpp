//
// key value的数组
//
//
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>


extern uint64_t CityHash64(const char *s, size_t len);

using namespace std;

class SimpleHash {
  private:
	uint64_t usedslots = 0;//观察实际使用水位，水位过半后hash冲突会增加，超过70%需要扩容

	vector<array<char, 64>> keys;
	vector<array<float, 64>> vals;
	vector<uint8_t> used;
	
	//If the key values are already uniformly distributed, using a hash gains us
	//nothing
	uint64_t hash(const array<char, 64> key)
	{
		string str(key.data());
		return CityHash64(str.c_str(), str.size());
	}

	bool isUsed(const uint64_t loc)
	{
		const auto used_loc = loc/8;
		const auto used_bit = 1<<(loc%8);
		return used[used_loc]&used_bit;    
	}

	void setUnused(const uint64_t loc)
	{
		const auto used_loc = loc/8;
		const auto used_bit = 1<<(loc%8);
		used[used_loc] = used[used_loc] &~ used_bit;
	}

	void setUsed(const uint64_t loc)
	{
		const auto used_loc = loc/8;
		const auto used_bit = 1<<(loc%8);
		used[used_loc] |= used_bit;
	}
  public:
	//size0 should be a prime and about 30% larger than the maximum number needed
	SimpleHash(uint64_t size0)
	{
		vals.resize(size0);
		keys.resize(size0);
		used.resize(size0/8+1,0);
	}

	SimpleHash()//空构造函数，用于从文件中加载
	{
	}

	void insert(const array<char, 64> key, const array<float, 64> val)
	{
		uint64_t loc = hash(key)%keys.size();

		//Use linear probing. Can create infinite loops if table too full.
		while(isUsed(loc)){ loc = (loc+1)%keys.size(); }

		setUsed(loc);
		usedslots++;
		keys[loc] = key;
		vals[loc] = val;
	}

	int remove(const array<char, 64> key)
	{
		uint64_t loc = hash(key)%keys.size();

		while(true)
		{
			if(!isUsed(loc))
			{
			  return -1;
			}  
			if(strcmp(keys[loc].data(), key.data()) == 0)//isUsed设置true 且 key相同情况下，证明找到了
			{
				//memset(keys[loc].data(), 0, sizeof(keys[loc][0]) * keys[loc].size());没有必要重置
				//memset(vals[loc].data(), 0, sizeof(vals[loc][0]) * vals[loc].size());
				setUnused(loc);
				usedslots--;
				return 0;
			}
			loc = (loc+1)%keys.size();
		}
	}

	int get(const array<char, 64> key, array<float, 64>& value)
	{
		uint64_t loc = hash(key)%keys.size();
		while(true)
		{
			if(!isUsed(loc))
			{
				return -1;
			}  
			if(strcmp(keys[loc].data(), key.data()) == 0)
			{
				value = vals[loc];
				return 0;
			}
			loc = (loc+1)%keys.size();
		}
	}

	uint64_t size() const
	{
		return keys.size();
	}

	uint64_t usedSize() const
	{
		return usedslots;
	}
	
	int tofile(string file_path)
	{
		FILE *f = fopen(file_path.c_str(), "wb+");
		if(f == NULL)
		{
		   cout << strerror(errno) << endl;
		   return -1;
		}
		uint64_t size = this->size();
		fwrite(&size, 8, 1, f); //前八个字节是table的size

		for(int i = 0; i < size; i++)
		{
			fwrite(keys[i].data(), 64,  1, f);  //64*char
		}
		for(int i = 0; i < size; i++)
		{
			fwrite(vals[i].data(), 256,  1, f);  //64*char
		}
		fwrite(used.data(), 1, size/8+1, f);
		fclose(f);
		return 0;
	}
	
	int loadfile(string file_path)
	{
		FILE *f = fopen(file_path.c_str(), "rb");
		if(f == NULL)
		{
		    cout << strerror(errno) << endl;
		    return -1;
		}

		uint64_t size;
		fread(&size, 8, 1, f);//前八个字节是table的size

		vals.resize(size);
		keys.resize(size);
		used.resize(size/8+1,0);

		for(int i = 0; i < size; i++)
		{
			fread(keys[i].data(), 64, 1, f);  //64*char
		}
		for(int i = 0; i < size; i++)
		{
			fread(vals[i].data(), 256, 1, f);  //64*float
		}
		fread(used.data(), 1, size/8+1, f);
		
		fclose(f);
		return 0;
	}
};

#define db_file "test.db"
#define FACE_DNA_KEY "face_dna_key"
const int TEST_TABLE_SIZE = 10000;

int main()
{
	//Perfectly horrendous way of seeding a PRNG, but we'll do it here for brevity
	auto generator = mt19937(12345); //Combination of my luggage
	//Generate values within the specified closed intervals

	SimpleHash map(1.38 * TEST_TABLE_SIZE);//1.38倍的hash容量
	cout<< "Created table of size "<< map.size()<<endl;

	cout<<"Generating test data..."<<endl;
//=========================测试insert接口==================================================
	for(int i = 0; i < TEST_TABLE_SIZE; i++)
	{
		array<char, 64> key = {0};
		sprintf(key.data(), "%s_%d", FACE_DNA_KEY, i);

		array<float, 64> value;
		for(int i = 0; i < 64; ++i)
		{
			uniform_real_distribution<float> dist(0.0, 9.9);
			value[i] = dist(generator);
		}

		map.insert(key, value); //Low chance of collisions, so we get quite close to the desired size
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
	array<char, 64> test_key = {"face_dna_key_777"};
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
		SimpleHash newmap;//测试空map从文件加载
		const auto start = chrono::steady_clock::now();
		newmap.loadfile(db_file);
		const auto end = chrono::steady_clock::now();
		cout<<"New Load time = "<< chrono::duration<double, milli> (end-start).count() << " ms" << endl;

		array<char, 64> test_key = {"face_dna_key_777"};
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