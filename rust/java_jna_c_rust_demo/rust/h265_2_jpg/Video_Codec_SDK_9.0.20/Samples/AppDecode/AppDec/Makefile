################################################################################
#
# Copyright 2017-2018 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
#
################################################################################

include ../../common.mk

LDFLAGS += -lnvcuvid
LDFLAGS += $(shell pkg-config --libs libavcodec libavutil libavformat)

INCLUDES += $(shell pkg-config --cflags libavcodec libavutil libavformat)

# Target rules
all: build

build: AppDec

NvDecoder.o: ../../NvCodec/NvDecoder/NvDecoder.cpp ../../NvCodec/NvDecoder/NvDecoder.h
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

AppDec.o: AppDec.cpp ../../NvCodec/NvDecoder/NvDecoder.h \
          ../../Utils/NvCodecUtils.h ../../Utils/Logger.h
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

AppDec: AppDec.o NvDecoder.o
	$(GCC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf AppDec AppDec.o NvDecoder.o
