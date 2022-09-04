#!/bin/bash

crate=$1

echo "Publishing ${crate} ..."
cd ${crate}
cargo publish --token ${CRATES_IO_API_TOKEN} || exit 1
echo "Sucessfully published ${crate}"
