typedef void(*Example22Callback)(int);
void example22_triggerCallback(const Example22Callback pfn)
{
	(*pfn)(230);
}
