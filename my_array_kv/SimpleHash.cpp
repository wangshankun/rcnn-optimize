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

const int TEST_TABLE_SIZE = 10000;
extern uint64_t CityHash64(const char *s, size_t len);

template<class K, class V>//std::array<char, 64>, std::array<float, 64>
class SimpleHash {
  public://这里为了调试方便这里设置为public,工程使用时候设置为private
	int usedslots = 0;//观察实际使用水位，水位过半后hash冲突会增加，超过70%需要扩容

	std::vector<K> keys;
	std::vector<V> vals;
	std::vector<uint8_t> used;

	//size0 should be a prime and about 30% larger than the maximum number needed
	SimpleHash(int size0)
	{
		vals.resize(size0);
		keys.resize(size0);
		used.resize(size0/8+1,0);
	}

	//If the key values are already uniformly distributed, using a hash gains us
	//nothing
	uint64_t hash(const K key)
	{
		std::string str(key.data());
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
	void insert(const K key, const V val)
	{
		uint64_t loc = hash(key)%keys.size();

		//Use linear probing. Can create infinite loops if table too full.
		while(isUsed(loc)){ loc = (loc+1)%keys.size(); }

		setUsed(loc);
		usedslots++;
		keys[loc] = key;
		vals[loc] = val;
	}

	int remove(const K key)
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
				memset(keys[loc].data(), 0, sizeof(keys[loc][0]) * keys[loc].size());
				memset(vals[loc].data(), 0, sizeof(vals[loc][0]) * vals[loc].size());
				setUnused(loc);
				usedslots--;
				return 0;
			}
			loc = (loc+1)%keys.size();
		}
	}

	int get(const K key, V& value)
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
};

#define db_file "test.db"

typedef SimpleHash<std::array<char, 64>, std::array<float, 64> > table_t;

void SaveSimpleHash(const table_t &map)
{
  std::cout<<"Save. ";
  const auto start = std::chrono::steady_clock::now();
  FILE *f = fopen(db_file, "wb+");
  if(f == NULL)
  {
	   std::cout << strerror(errno) << std::endl;
  }
  uint64_t size = map.size();
  fwrite(&size, 8, 1, f);
  for(int i = 0; i < size; i++)
  {
  	fwrite(map.keys[i].data(), 64,  1, f);  //64*char
  }
  for(int i = 0; i < size; i++)
  {
  	fwrite(map.vals[i].data(), 256,  1, f);  //64*char
  }
  fwrite(map.used.data(), 1, size/8+1, f);
  fclose(f);
  const auto end = std::chrono::steady_clock::now();
  std::cout<<"Save time = "<< std::chrono::duration<double, std::milli> (end-start).count() << " ms" << std::endl;
}

table_t LoadSimpleHash()
{
  std::cout<<"Load. ";
  const auto start = std::chrono::steady_clock::now();
  FILE *f = fopen(db_file, "rb+");

  uint64_t size;
  fread(&size, 8, 1, f);

  table_t map(size);

  for(int i = 0; i < size; i++)
  {
  	fread(map.keys[i].data(), 64, 1, f);  //64*char
  }
  for(int i = 0; i < size; i++)
  {
  	fread(map.vals[i].data(), 256, 1, f);  //64*float
  }
  fread(map.used.data(), 1, size/8+1, f);
  fclose(f);
  const auto end = std::chrono::steady_clock::now();
  std::cout<<"Load time = "<< std::chrono::duration<double, std::milli> (end-start).count() << " ms" << std::endl;

  return map;
}

int main()
{
  //Perfectly horrendous way of seeding a PRNG, but we'll do it here for brevity
  auto generator = std::mt19937(12345); //Combination of my luggage
  //Generate values within the specified closed intervals

  #define FACE_DNA_KEY "face_dna_key"

  table_t map(1.38 * TEST_TABLE_SIZE);//1.38倍的hash容量
  std::cout<<"Created table of size "<<map.size()<<std::endl;

  std::cout<<"Generating test data..."<<std::endl;
  for(int i = 0;i < TEST_TABLE_SIZE; i++)
  {
	std::array<char, 64> key = {0};
	sprintf(key.data(), "%s_%d", FACE_DNA_KEY, i);
	
    std::array<float, 64> value;
	for(int i = 0; i < 64; ++i)
	{
		std::uniform_real_distribution<float> dist(0.0, 9.9);
		value[i] = dist(generator);
	}

    map.insert(key, value); //Low chance of collisions, so we get quite close to the desired size
  }

  SaveSimpleHash(map);
  auto newmap = LoadSimpleHash();

//=========================测试从文件读出来的数据是否正确==================================================
  for(int i = 0; i< map.keys.size(); i++)
  {
	for(int j = 0; j < 64; j++)
	{
		assert(map.keys.at(i).at(j)==newmap.keys.at(i).at(j)); 
	}
  }

  for(int i = 0; i < map.vals.size(); i++)
  {
	for(int j = 0; j < 64; j++)
	{
		assert(map.vals.at(i).at(j)==newmap.vals.at(i).at(j)); 
	}
  }

  for(int i=0;i<map.used.size();i++)
  {
	assert(map.used.at(i)==newmap.used.at(i));    
  }

//=========================打印一组数据内部信息==================================================
  printf("map.keys.at(10):%s  newmap.keys.at(10):%s\r\n",map.keys.at(10).data(), newmap.keys.at(10).data());

  printf("map.vals.at(10):\r\n");
  for(int j = 0; j < 64; j++)
  {
	if(j%8 == 0 && j != 0) printf("\r\n");
  	printf("%.8f ", map.vals.at(10).at(j));
  }
  printf("\r\n");
  printf("newmap.vals.at(10):\r\n");
  for(int j = 0; j < 64; j++)
  {
	if(j%8 == 0 && j != 0) printf("\r\n");
  	printf("%.8f ", newmap.vals.at(10).at(j));
  }  
  printf("\r\n");
  
//=========================测试get接口==================================================
  std::array<char, 64>  test_key = map.keys.at(10);
  std::array<float, 64> test_value = {0};
  if(map.get(test_key, test_value) == 0)
  {
	  printf("get map.keys.at(10) 's value:\r\n");
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
//=========================测试remove接口==================================================
  if(map.remove(test_key) == 0)//先删除
  {
  }
  else
  {
	std::cout<<"Remove Failed, Not Found"<<std::endl;
  }
  
  if(map.get(test_key, test_value) == 0)//再查找
  {

  }
  else
  {
	std::cout<<"Get Failed, Not Found"<<std::endl;
  }
  
  printf("map.vals.at(10):\r\n");//使用内部接口读出删除key的内存，是否被重置为0; 这里为了调试方便把keys和vals设置为public,工程使用时候设置为private
  for(int j = 0; j < 64; j++)
  {
	if(j%8 == 0 && j != 0) printf("\r\n");
  	printf("%.8f ", map.vals.at(10).at(j));
  }
  printf("\r\n");
}