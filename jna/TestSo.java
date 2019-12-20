import com.sun.jna.Library;
import com.sun.jna.Native;
public class TestSo {
public interface LgetLib extends Library {
// 调用linux下面的so文件,注意，这里只要写test就可以了，不要写libtest，也不要加后缀
LgetLib INSTANCE = (LgetLib) Native.loadLibrary("test",LgetLib.class);
int add(int a,int b);
}
public int add(int a,int b){
return LgetLib.INSTANCE.add(a,b);
}
public static void main(String[] args) {
TestSo ts = new TestSo();
int c = ts.add(10,20);
System.out.println("10+20="+c);
}
}
