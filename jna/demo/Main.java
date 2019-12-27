///package com.alibaba.damo.shic;
import java.util.Arrays;
import java.util.List;

import com.sun.jna.Callback;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.Memory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class Main {
    public static interface CLibrary extends Library {

    public class CompressInputImage extends Structure {

            @Override
            protected List<String> getFieldOrder() {
                return Arrays.asList(
                    "image_id", "channel_id", "ts_ms", "compress_rate", "image_format", "buf", "buf_len");
            }

            public static class ByReference extends CompressInputImage implements Structure.ByReference {};
            public static class ByValue extends CompressInputImage implements Structure.ByValue {};

            public String  image_id;        /// 图像的id
            public String  channel_id;    /// 图像的channel id
            public long    ts_ms;            /// int64 时间戳
            public int     compress_rate;    ///  压缩率： 0 不压缩， 100 最大压缩比。
            public int     image_format;     ///  图像的输入格式。 1: jpg, 2: png
            public Pointer buf;              /// 这个压缩的byte array的buffer, 这个地址通过Memory分配的, 由调用者管理
            public long    buf_len;          /// 上面这个buffer的字节数长度
        }

    public class CompressOutputData extends Structure {

            public static class ByReference extends  CompressOutputData implements Structure.ByReference {};
            public static class ByValue extends CompressOutputData implements Structure.ByValue {};
            
            @Override
            protected List<String> getFieldOrder() {
            return Arrays.asList("version","channel_id","image_ids","ts_array",
                    "ts_array_len", "compressed_buf", "compressed_buf_len");
            }

            public int     version;
            public String  channel_id;
            public String  image_ids;
            public Pointer ts_array;
            public int     ts_array_len;
            public Pointer compressed_buf;
            public long    compressed_buf_len;           /// content buffer的长度
            
        }

        public interface CompressResultCallback extends Callback {
            void invoke(int resultCode, CompressOutputData.ByReference oneResult);
        }
        public static class CompressResultCallbackImplementation implements CompressResultCallback {
            @Override
            public void invoke(int resultCode, CompressOutputData.ByReference oneResult) {
                    System.out.println("CompressOutputData callback resultCode: " + resultCode);
                    System.out.println("version:" + oneResult.version + "  image_ids:" + oneResult.image_ids);
                    System.out.println("compressed_buf_len:" + oneResult.compressed_buf_len);
                    /*
                    //读出压缩好的内存，存成文件，image_ids，channel_id信息存到对应数据库中
                    byte[] buf = new byte[oneResult.compressed_buf_len];
                    System.out.println("one_data: " + one_data);
                    oneResult.compressed_buf.read(0,buf,0,oneResult.compressed_buf_len);
                    for(byte x : buf ){
                         System.out.println(x);
                    }*/
            }
        }

        public class DecompressInputData extends Structure {
            
            public static class ByReference extends  DecompressInputData implements Structure.ByReference {};
            public static class ByValue extends DecompressInputData implements Structure.ByValue {};
            
            @Override
            protected List<String> getFieldOrder() {
                return Arrays.asList("version", "channel_id", "ts_array", "ts_array_len",
                    "image_ids", "compressed_buf", "compressed_buf_len",
                    "target_indexs", "image_compress_rate", "image_compress_method");
            }
            public int     version;
            public String  channel_id;
            public Pointer ts_array;
            public long    ts_array_len;
            public String  image_ids;
            public Pointer compressed_buf;
            public long    compressed_buf_len;
            public String  target_indexs;
            public int     image_compress_rate;
            public int     image_compress_method;
        }

        public class DecompressoOutput extends Structure {
            public static class ByReference extends  DecompressoOutput implements Structure.ByReference {};
            public static class ByValue extends DecompressoOutput implements Structure.ByValue {};
            
            public String  channel_id;
            public String  image_id;
            public long    ts_ms;
            public Pointer image_buf;
            public long    image_buf_len;
            public int     image_format;
            @Override
            protected List getFieldOrder() {
                return Arrays.asList("channel_id", "image_id", "ts_ms", "image_buf", "image_buf_len", "image_format");
            }
        }

        public interface DecompressResultCallback extends Callback {
            void invoke(int resultCode, DecompressoOutput.ByReference oneResult);
        }
        public static class DecompressResultCallbackImplementation implements DecompressResultCallback {
            @Override
            public void invoke(int resultCode, DecompressoOutput.ByReference oneResult) {
                System.out.println("DecompressoOutput callback resultCode");
            }
        }

        int  init(boolean enable_gpu, int gpu_id, boolean enable_cpu_bind, int[] cpu_ids, int cpu_ids_len);
        void compressListOfImage(CompressInputImage.ByReference input_images, int len, CompressResultCallback callback);
        void decompressListOfImage(DecompressInputData.ByReference input_infos, int len, DecompressResultCallback callback);
    }

    public static void main(String[] args)
    {
        final CLibrary INSTANCE = (CLibrary)Native.loadLibrary("SHIClib",CLibrary.class);
       
        //初始化函数
        INSTANCE.init(true,0,false,null,0);

        //压缩例子
        //一次请求压缩9张图
        int comp_num  = 9;
        final CLibrary.CompressResultCallbackImplementation CompressCallbackImpl = new CLibrary.CompressResultCallbackImplementation();
        final CLibrary.CompressInputImage.ByReference arrays_ref = new CLibrary.CompressInputImage.ByReference();
        final CLibrary.CompressInputImage[] arrays = (CLibrary.CompressInputImage[])arrays_ref.toArray(comp_num);
        
        int ts_ms = 1234560;//假设时间戳开始日期
        int count = 100;

        for(CLibrary.CompressInputImage array : arrays )
        {
            Memory memory = new Memory(3*1024*1024 * Native.getNativeSize(byte.class));//一张图最大3M
            byte[] img_buf = memory.getByteArray(0, 3*1024*1024);
            try
            {   
                
                String count_str = Integer.toString(count);
                String file_name = count_str + ".jpg";
                InputStream input = new FileInputStream(file_name);
                int bytesRead = 0;
                bytesRead = input.read(img_buf);
                //System.out.println(bytesRead);
                memory.write(0, img_buf, 0, bytesRead);
                
                array.image_id       = count_str;
                if (count < 106)//前6张是一个channel
                {
                    array.channel_id   = "hanzhou_1837_12111"; 
                }
                else//后三张是一个channel
                {
                    array.channel_id   = "hanzhou_1137_12121"; 
                }
                array.ts_ms          = ts_ms;
                array.compress_rate  = 16; 
                array.image_format   = 0;  
                array.buf            = memory.share(0, bytesRead);
                array.buf_len        = bytesRead;

                ts_ms   = ts_ms + 100;
                count   = count + 1;
            }
            catch(IOException e)
            {
                System.out.println(e.getMessage());
            }
        }
        //调用压缩函数
        INSTANCE.compressListOfImage(arrays_ref, comp_num, CompressCallbackImpl);


        //解压例子:数据是上面压缩的两个包
        //总计两个包裹（目前例子按照一个channel一个包来）
        int pkg_num = 2;

        final CLibrary.DecompressResultCallbackImplementation DecompressCallbackImpl = new CLibrary.DecompressResultCallbackImplementation();
        final CLibrary.DecompressInputData.ByReference dci_e6ref = new CLibrary.DecompressInputData.ByReference();
        final CLibrary.DecompressInputData[] darrays = (CLibrary.DecompressInputData[])dci_e6ref.toArray(pkg_num);
        
        //包裹0总计6张图，需要解压其中的0,2,4 
        int pkg0_img_num = 6;
        int pkg0_len     = 3227097;//包裹0 字节总长度
        Memory pkg0_ts_array_mem  = new Memory(pkg0_img_num * Native.getNativeSize(Long.class));
        pkg0_ts_array_mem.setLong(0, 1234560);
        pkg0_ts_array_mem.setLong(1, 1234660);
        pkg0_ts_array_mem.setLong(2, 1234760);
        pkg0_ts_array_mem.setLong(3, 1234860);
        pkg0_ts_array_mem.setLong(4, 1234960);
        pkg0_ts_array_mem.setLong(5, 1235060);
        Memory pkg0_buf_mem       = new Memory(pkg0_len * Native.getNativeSize(byte.class));;
        darrays[0].version               = 0;
        darrays[0].channel_id            = "hanzhou_1837_12111"; 
        darrays[0].ts_array              = pkg0_ts_array_mem.share(0);
        darrays[0].ts_array_len          = pkg0_img_num; 
        darrays[0].image_ids             = "100;101;102;103;104;105;";
        darrays[0].compressed_buf        = pkg0_buf_mem.share(0);
        darrays[0].compressed_buf_len    = pkg0_len;
        darrays[0].target_indexs         = "0;2;4";
        darrays[0].image_compress_rate   = 12;
        darrays[0].image_compress_method = 0;

        //包裹1总计3张图，需要解压其中的2
        int pkg1_img_num = 3;
        int pkg1_len     = 1693714;//包裹1 字节总长度
        Memory pkg1_ts_array_mem  = new Memory(pkg1_img_num * Native.getNativeSize(Long.class));
        pkg1_ts_array_mem.setLong(0, 1235160);
        pkg1_ts_array_mem.setLong(1, 1235260);
        pkg1_ts_array_mem.setLong(2, 1235360);
        Memory pkg1_buf_mem       = new Memory(pkg1_len * Native.getNativeSize(byte.class));;
        darrays[1].version               = 0;
        darrays[1].channel_id            = "hanzhou_1137_12121"; 
        darrays[1].ts_array              = pkg1_ts_array_mem.share(0);
        darrays[1].ts_array_len          = pkg1_img_num; 
        darrays[1].image_ids             = "106;107;108;";
        darrays[1].compressed_buf        = pkg1_buf_mem.share(0);
        darrays[1].compressed_buf_len    = pkg1_len;
        darrays[1].target_indexs         = "2";
        darrays[1].image_compress_rate   = 12;
        darrays[1].image_compress_method = 0;

        INSTANCE.decompressListOfImage(dci_e6ref, pkg_num, DecompressCallbackImpl);

    }
}
