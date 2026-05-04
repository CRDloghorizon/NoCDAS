rm -rf build
mkdir build && cd build
cmake ..
make
cd ..
./build/NoCDASim
rm -rf build