import java.util.Arrays;
import java.util.List;

import com.sun.jna.Callback;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;

public class TestSo {

public interface CLibrary extends Library {
	// define an interface that wraps the callback code
	public interface Example22CallbackInterface extends Callback {
		void invoke(int val);
	}

	// define an implementation of the callback interface
	public static class Example22CallbackImplementation implements Example22CallbackInterface {
		@Override
		public void invoke(int val) {
			System.out.println("example22: " + val);
		}
	}

	public void example22_triggerCallback(Example22CallbackInterface callback);
}

public static void main(String[] args) {
final CLibrary clib = (CLibrary)Native.loadLibrary("testlib", CLibrary.class);
final CLibrary.Example22CallbackImplementation callbackImpl = new CLibrary.Example22CallbackImplementation();
clib.example22_triggerCallback(callbackImpl);
}

}
