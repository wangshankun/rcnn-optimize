

conan create . 20210730@hci/stable -pr:h conanprofile.ingenic_t40_uclibc -pr:b conanprofile.x86_64  -b missing

conan upload ingenic-venus/20210730@hci/stable -r hci

