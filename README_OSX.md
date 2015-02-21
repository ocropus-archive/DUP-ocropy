# Patch for OSX (clang)

The compile flag `-fopenmp` doesn't work for clang,
so if your `gcc` is link to `clang` you might need to take  solutions below:

1. Install `gcc` and make sure `gcc` in `$PATH` is point to real `gcc`,
   not `clang`.
2. If you don't have `gcc` installed, apply patch below to change compile flag.

``` diff
--- ocrolib/native.py	2015-02-21 13:34:52.000000000 +0800
+++ ocrolib_osx/native.py	2015-02-21 15:29:56.000000000 +0800
@@ -41,7 +41,7 @@
     pass
 
 def compile_and_find(c_string,prefix=".pynative",opt="-g -O4",libs="-lm",
-                     options="-shared -fopenmp -std=c99 -fPIC",verbose=0):
+                     options="-shared -openmp -std=c99 -fPIC",verbose=0):
     if not os.path.exists(prefix):
         os.mkdir(prefix)
     m = hashlib.md5()
```
