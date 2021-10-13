#include "hashmap_db.h"
extern uint64_t CityHash64(const char *s, size_t len);

//已经做过l2_norm可以用cos_distance_unit_id计算余弦距离
float SimpleHash::cos_distance_unit_id(const array<float, 64> &val_a, const array<float, 64> &val_b)
{
	float dist = 0;
	for (int i = 0; i < 64; i++)
	{
		dist += ((val_a[i]) * (val_b[i]));
	}
	return dist;
}
//计算两vector的余弦距离
float SimpleHash::cos_distance(const array<float, 64> &val_a, const array<float, 64> &val_b)
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

uint64_t SimpleHash::hash(const string key)
{
	return CityHash64(key.c_str(), key.size());
}

bool SimpleHash::isUsed(const uint64_t loc)
{
	const auto used_loc = loc/8;
	const auto used_bit = 1<<(loc%8);
	return used[used_loc]&used_bit;    
}

void SimpleHash::setUnused(const uint64_t loc)
{
	const auto used_loc = loc/8;
	const auto used_bit = 1<<(loc%8);
	used[used_loc] = used[used_loc] &~ used_bit;
}

void SimpleHash::setUsed(const uint64_t loc)
{
	const auto used_loc = loc/8;
	const auto used_bit = 1<<(loc%8);
	used[used_loc] |= used_bit;
}

SimpleHash::SimpleHash(uint64_t size0)
{
	vals.resize(size0);
	keys.resize(size0);
	used.resize(size0/8+1,0);
}

SimpleHash::SimpleHash()//空构造函数，用于从文件中加载
{
}

SimpleHash::SimpleHash(SimpleHash &obj)//拷贝构造
{
	uint64_t size0 = obj.size();
	vals = obj.vals;
	keys = obj.keys ;
	used = obj.used;
}

void SimpleHash::get_all_keys(vector<string> &key_vec)
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

string SimpleHash::from_similarvalue_get_key(const array<float, 64> &val, float th_hold)
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

int SimpleHash::insert(const string key, const array<float, 64> &val)
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

int SimpleHash::remove(const string key)
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

int SimpleHash::get(const string key, array<float, 64>& value)
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

uint64_t SimpleHash::size()
{
	return keys.size();
}

uint64_t SimpleHash::used_size()
{
	return usedslots;
}

int SimpleHash::tofile(string file_path)
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

int SimpleHash::loadfile(string file_path)
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
