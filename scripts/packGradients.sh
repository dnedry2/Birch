pushd ../data
rm ../bin/gradients.bpk
./../bin/packer -p gradients.bpk Gradients/
mv gradients.bpk ../bin/
popd
