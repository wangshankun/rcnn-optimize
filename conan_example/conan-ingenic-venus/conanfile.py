from conans import ConanFile, tools
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

class CommonConan(ConanFile):#类名无所谓
    name = "ingenic-venus"#name变量要设置对
    license = ""
    url = ""
    description = "venus engine from ingenic"
    settings = {
        "os": ["Linux"],
        "arch": ["mips"],
        "compiler": {###这些配置在create命令传入的conanprofile里面setting设置好
            "gcc": {
                "vendor": ["ingenic"],
                "version": ["7.2"],
                "libc": ["glibc", "uclibc"],
            }
        },
    }

    def requirements(self):
        pass

    @property
    def _source_subfolder(self):
        return "source_subfolder"

    def source(self):
        self.output.info("version : %s" % self.version)
        self.output.info("source from : %s" % self.conan_data["sources"][self.version])
        tools.get(**self.conan_data["sources"][self.version])#conan_data就是conandata.yml,version是create命令时候传入
        extracted_dir = self.version
        os.rename(extracted_dir, self._source_subfolder)

    def build(self):
        pass

    def package(self):
        if self.settings.compiler.version == "7.2":
            compiler_version = "7.2.0"

        self.copy(
            pattern="*",
            src="{}/{}/include".format(self._source_subfolder, compiler_version),
            dst="include",
            keep_path=False,
        )
        self.copy(
            pattern="*",
            src="{}/{}/lib/{}".format(
                self._source_subfolder,
                compiler_version,
                str(self.settings.compiler.libc),
            ),
            dst="lib",
            keep_path=False,
        )

    def package_info(self):
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.libdirs = ["lib"]
        self.cpp_info.libs = ["libvenus.d.so", "libvenus.p.so", "libvenus.so"]
