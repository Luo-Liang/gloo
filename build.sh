echo "building..."
mkdir -p build
git pull
cd build
rm * -rf
cmake .. -DUSE_CUDA=1 -DUSE_REDIS=1 -DBUILD_SHARED_LIBS=ON -DUSE_IBVERBS=0 -DBUILD_BENCHMARK=1
make -j32
