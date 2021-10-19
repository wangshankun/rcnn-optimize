

conan create . 20210730@hci/stable -pr:h conanprofile.ingenic_t40_uclibc -pr:b conanprofile.x86_64  -b missing

conan upload ingenic-venus/20210730@hci/stable -r hci


```
如果更新20210730@hci/stable 这个版本, 那么重新create，upload一次，覆盖原来的;
```
