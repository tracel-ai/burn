echo "Setup arrayfire backend"
export AF_PATH=$HOME/.local/share/arrayfire
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AF_PATH/lib64

echo "Setup tch backend"
export LIBTORCH=${HOME}/.local/lib/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
