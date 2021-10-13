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
using namespace std;

class SimpleHash {
  private:
	uint64_t usedslots = 0;//观察实际使用水位，水位过半后hash冲突会增加，超过70%需要扩容

	vector<array<char, 64>> keys;
	vector<array<float, 64>> vals;
	vector<uint8_t> used;

	//已经做过l2_norm可以用cos_distance_unit_id计算余弦距离
	float cos_distance_unit_id(const array<float, 64> &val_a, const array<float, 64> &val_b);
	//计算两vector的余弦距离
	float cos_distance(const array<float, 64> &val_a, const array<float, 64> &val_b);

	//If the key values are already uniformly distributed, using a hash gains us
	//nothing
	uint64_t hash(const string key);
	bool isUsed(const uint64_t loc);
	void setUnused(const uint64_t loc);
	void setUsed(const uint64_t loc);
  public:
	//size0 should be a prime and about 30% larger than the maximum number needed
	SimpleHash(uint64_t size0);
	SimpleHash();
 	SimpleHash(SimpleHash &obj);
	
        void get_all_keys(vector<string> &key_vec);
	string from_similarvalue_get_key(const array<float, 64> &val, float th_hold);
	int insert(const string key, const array<float, 64> &val);
	int remove(const string key);
	int get(const string key, array<float, 64>& value);
	uint64_t size();
	uint64_t used_size();
	int tofile(string file_path);
	int loadfile(string file_path);
};
