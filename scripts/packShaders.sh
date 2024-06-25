rm bin/shaders.bpk
pushd apps/bsat/
./../../bin/packer -p shaders.bpk shaders/
mv shaders.bpk ../../bin/
popd