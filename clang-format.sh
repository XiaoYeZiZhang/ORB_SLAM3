#! /bin/bash
    function read_dir(){
        # shellcheck disable=SC2045
        for file in `ls $1`
        do
            if [ -d $1"/"$file ]
            then
                read_dir $1"/"$file
            elif [ "${file##*.}"x = "cc"x ]||[ "${file##*.}"x = "h"x ]
            then
                #echo $1"/"$file
                `clang-format -i $1"/"$file`
            fi
        done
    }
    #读取第一个参数
    read_dir $1