pushd ../data
rm ../bin/theme.bpk
./../bin/packer -p theme.bpk bicons/
mv theme.bpk ../bin/
popd
