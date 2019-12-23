import java.util.Arrays;
import java.util.List;

import com.sun.jna.Callback;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;

public class TestSo {

public interface CLibrary extends Library {
	public static class Example4Struct extends Structure {
                public static class ByValue extends Example4Struct implements Structure.ByValue {}
		public int val;

                protected List<String> getFieldOrder() {
                    return Arrays.asList("val");
                }
	}
        public Example4Struct.ByValue example4_getStruct();
}

public static void main(String[] args) {
final CLibrary clib = (CLibrary)Native.loadLibrary("testlib", CLibrary.class);
final CLibrary.Example4Struct.ByValue e4val = clib.example4_getStruct();
System.out.println("example 4: " + e4val.val);
}

}
