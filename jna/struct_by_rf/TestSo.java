import java.util.Arrays;
import java.util.List;

import com.sun.jna.Callback;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;

public class TestSo {

public interface CLibrary extends Library {
	public static class Example3Struct extends Structure {
		public static class ByReference extends Example3Struct implements Structure.ByReference {}

		public int val;

                protected List<String> getFieldOrder() {
                    return Arrays.asList("val");
                }
	}
	public void example3_sendStruct(Example3Struct.ByReference sval);

}

public static void main(String[] args) {
final CLibrary clib = (CLibrary)Native.loadLibrary("testlib", CLibrary.class);
final CLibrary.Example3Struct.ByReference e3ref = new CLibrary.Example3Struct.ByReference();
e3ref.val = 7;
clib.example3_sendStruct(e3ref);
}

}
