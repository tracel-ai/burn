$Env:LIBTORCH += ";$(pwd)/libtorch/"
$Env:Path += ";$(pwd)/libtorch/;$(pwd)/libtorch/lib/"
./train.exe
