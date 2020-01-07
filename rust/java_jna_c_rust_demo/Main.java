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
import java.io.OutputStream;

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
                    "image_id", "channel_id", "ts_ms", "buf", "buf_len");
            }

            public static class ByReference extends CompressInputImage implements Structure.ByReference {};
            public static class ByValue extends CompressInputImage implements Structure.ByValue {};

            public String  image_id;        /// 图像的id
            public String  channel_id;    /// 图像的channel id
            public long    ts_ms;            /// int64 时间戳
            public Pointer buf;              /// 这个压缩的byte array的buffer, 这个地址通过Memory分配的, 由调用者管理
            public long    buf_len;          /// 上面这个buffer的字节数长度
        }

    public class CompressOutputData extends Structure {

            public static class ByReference extends  CompressOutputData implements Structure.ByReference {};
            public static class ByValue extends CompressOutputData implements Structure.ByValue {};
            
            @Override
            protected List<String> getFieldOrder() {
            return Arrays.asList("channel_ids", "image_ids", "ts_arrays",
                                 "offsets", "version", "compress_rate",
                                 "image_format", "compressed_buf", "compressed_buf_len");
            }
            public String          channel_ids;
            public String          image_ids;
            public String          ts_arrays;
            public String          offsets;
            public int             version;
            public int             compress_rate;
            public int             image_format;
            public Pointer         compressed_buf;
            public long            compressed_buf_len;
        }

        public interface CompressResultCallback extends Callback {
            void invoke(int resultCode, CompressOutputData.ByReference oneResult);
        }
        public static class CompressResultCallbackImplementation implements CompressResultCallback {
            @Override
            public void invoke(int resultCode, CompressOutputData.ByReference oneResult) {
                    //System.out.println("CompressOutputData callback resultCode: " + resultCode);
                    //System.out.println("version:" + oneResult.version + "  image_ids:" + oneResult.image_ids);
                    //System.out.println("compressed_buf_len:" + oneResult.compressed_buf_len);

                    System.out.println("Compress callback");

                    String[] img_ids;
                    img_ids = oneResult.image_ids.split(";"); // 分割字符串 
                    String pkg_name = img_ids[0] + "_" + img_ids[img_ids.length - 1] + ".bin";

                    //读出压缩好的内存，存成文件，image_ids，channel_id信息存到对应数据库中
                    byte[] pak_head_len_ay = new byte[8];
                    oneResult.compressed_buf.read(0, pak_head_len_ay, 0, 8);//前8字节存储包头长度
                    long pak_head_len = ByteBuffer.wrap(pak_head_len_ay).order(ByteOrder.LITTLE_ENDIAN).getLong();
                    long pak_len = 8 + pak_head_len + oneResult.compressed_buf_len;//包总长度
                    ByteBuffer pak_data = oneResult.compressed_buf.getByteBuffer(0, pak_len);

                    writeByteArrayToFile(pkg_name, pak_data, pak_len);
            }
        }

        public interface DecompressResultCallback extends Callback {
            void invoke(int resultCode, CompressInputImage.ByReference oneResult);
        }
        public static class DecompressResultCallbackImplementation implements DecompressResultCallback {
            @Override
            public void invoke(int resultCode, CompressInputImage.ByReference oneResult) {
                    System.out.println("Decompress callback");
                    String img_name = oneResult.image_id + "_dcpress" + ".jpg";
                    long img_len = oneResult.buf_len;//图片长度
                    ByteBuffer img_data = oneResult.buf.getByteBuffer(0, img_len);
                    writeByteArrayToFile(img_name, img_data, img_len);
            }
        }

        int  init(boolean enable_gpu, int gpu_id, boolean enable_cpu_bind, int[] cpu_ids, int cpu_ids_len);
        void compressListOfImage(CompressInputImage.ByReference input_images, int len, CompressResultCallback callback);
        void decompressListOfImage(Pointer buf, int len, String hit_img_ids, DecompressResultCallback callback);
    }
    
    public static void writeByteArrayToFile(String fileName, ByteBuffer data, Long len) 
    {
        int len_i = len.intValue();
        byte[] data_byte = new byte[len_i];
        data.get(data_byte, 0, len_i);
        try
        {
            OutputStream output = new FileOutputStream(fileName);
            output.write(data_byte, 0, len_i);
            output.close();
        }
        catch(IOException e)
        {
            System.out.println(e.getMessage());
        }
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


        //解压例子
        Memory dec_memory = new Memory(16*1024*1024 * Native.getNativeSize(byte.class));//一个压缩包图最大16M
        byte[] dec_buf = dec_memory.getByteArray(0, 16*1024*1024);
        
        final CLibrary.DecompressResultCallbackImplementation DecompressCallbackImpl = new CLibrary.DecompressResultCallbackImplementation();
        try
        {
            InputStream dec_f = new FileInputStream("100_105.bin");
            int bytes_read = dec_f.read(dec_buf);
            dec_memory.write(0, dec_buf, 0, bytes_read);
            //从包里面取出101、104两张图片
            INSTANCE.decompressListOfImage(dec_memory.share(0, bytes_read), bytes_read, "101;104", DecompressCallbackImpl);
        }
        catch(IOException e)
        {
            System.out.println(e.getMessage());
        }
        
    }
}
