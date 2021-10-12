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

	//已经做过l2_norm可以用cos_distance_unit_id计算余弦距离
	float cos_distance_unit_id(const array<float, 64> &val_a, const array<float, 64> &val_b)
	{
		float dist = 0;
		for (int i = 0; i < 64; i++)
		{
			dist += ((val_a[i]) * (val_b[i]));
		}
		return dist;
	}
	//计算两vector的余弦距离
	float cos_distance(const array<float, 64> &val_a, const array<float, 64> &val_b)
	{
		float l2_norm_1 = 0;
		float l2_norm_2 = 0;
		float dist = 0;
		for (int i = 0; i < 64; i++)
		{
			l2_norm_1 += ((val_a[i]) * (val_a[i]));
			l2_norm_2 += ((val_b[i]) * (val_b[i]));
		}
		l2_norm_1 = sqrt(l2_norm_1);
		l2_norm_2 = sqrt(l2_norm_2);
		for (int i = 0; i < 64; i++)
		{
			dist += ((val_a[i]) * (val_b[i]) / (l2_norm_1 * l2_norm_2));
		}
		return dist;
	}

	//If the key values are already uniformly distributed, using a hash gains us
	//nothing
	uint64_t hash(const string key)
	{
		return CityHash64(key.c_str(), key.size());
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

    void get_all_keys(vector<string> &key_vec)
	{
		for(int i = 0; i < keys.size(); i++)
		{
			if(isUsed(i))
			{
				string str(keys[i].data());
				key_vec.push_back(str);
			}
		}
	}

	string from_similarvalue_get_key(const array<float, 64> &val, float th_hold)
	{
		for(int i = 0; i < keys.size(); i++)
		{
			if(isUsed(i))
			{
				float score = cos_distance(val, vals[i]);
                if(score >= th_hold)
                {
					string str(keys[i].data());
                    return str;
                }
			}
		}
		string str;
		return str;
	}

	int insert(const string key, const array<float, 64> &val)
	{
		if(usedslots >= keys.size() * 0.7)
		{
			printf("Error insert faild, db full \r\n");
			return -1;
		}
		if(key.length() >= 63)
		{
			printf("Error insert faild, key length is overflow\r\n");
			return -1;
		}

		uint64_t loc = hash(key)%keys.size();

		//Use linear probing. Can create infinite loops if table too full.
		while(isUsed(loc)){ loc = (loc+1)%keys.size(); }

		setUsed(loc);
		usedslots++;

		//keys[loc] = key;
		std::copy(key.begin(), key.end(), keys[loc].data());
		keys[loc][key.length()] = 0;
		vals[loc] = val;
	}

	int remove(const string key)
	{
		uint64_t loc = hash(key)%keys.size();

		while(true)
		{
			if(!isUsed(loc))
			{
			  return -1;
			}
			if(strcmp(keys[loc].data(), key.c_str()) == 0)//isUsed设置true 且 key相同情况下，证明找到了
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

	int get(const string key, array<float, 64>& value)
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

	uint64_t used_size() const
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
		SimpleHash newmap;//测试空map从文件加载
		const auto start = chrono::steady_clock::now();
		newmap.loadfile(db_file);
		const auto end = chrono::steady_clock::now();
		cout<<"New Load time = "<< chrono::duration<double, milli> (end-start).count() << " ms" << endl;

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
