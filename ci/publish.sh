#!/bin/bash

crate=$1

case ${crate} in
    burn-tensor)
        cd burn-tensor/
        echo "Publishing burn-tensor ..."
        ;;
    burn)
        echo "Publishing burn ..."
        ;;
    *)
        echo "Crate ${crate} unknown"
        exit 1
        ;;
esac

cargo publish --token ${CRATES_IO_API_TOKEN} || exit 1
echo "Sucessfully published ${crate}"
