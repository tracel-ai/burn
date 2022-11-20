#!/bin/bash

crate=${1}

find_value_inside_quotes() {
    text=${1}
    value=$(echo ${text} | cut -d'"' -f2)
    echo ${value}
}

current_crate_version() {
    value=""
    while IFS= read -r line; do
        case ${line} in "version"*)
            value=$(find_value_inside_quotes "${line}")
        esac
    done < "Cargo.toml"

    echo ${value}
}

publish() {
    echo "Publishing ${crate} ..."

    cd ${crate}
    version_local=$(current_crate_version)
    version_remote=$(remote_crate_version)

    echo "  - local version  = '${version_local}'"
    echo "  - remote version = '${version_remote}'"

    if [ "${version_local}" == "${version_remote}" ]; then
        echo "Remote version ${version_remote} is up to date, skipping deployment"
    else
        $(cargo publish --token ${CRATES_IO_API_TOKEN})
        rc=${?}

        if [[ ${rc} != 0 ]]; then
            echo "Fail to publish crate ${crate}"
            return 1
        fi

        echo "Sucessfully published ${crate} version ${version_local}"
        return 0
    fi
}

remote_crate_version() {
    text=$(cargo search ${crate} --limit 1)
    value=$(find_value_inside_quotes "${text}")
    echo ${value}
}

publish || exit 1
