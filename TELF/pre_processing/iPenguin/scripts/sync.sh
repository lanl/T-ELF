#!/bin/bash
VERBOSE=0

usage() {  # function to display usage and help message
    echo "Usage: $0 [-v] <source_directory> <target_directory>"
    echo
    echo "Copy files from source directory to target directory."
    echo "This script can be used for syncing file system iPenguin cache"
    echo "between multiple machines."
    echo
    echo "Options:"
    echo "  -v, --verbose       Turn on verbose mode"
    echo "  -h, --help          Display this help message"
    exit 1
}

# check for flags
while [ "$1" != "" ]; do
    case $1 in
        -v | --verbose )   VERBOSE=1
                            shift
                            ;;
        -h | --help )      usage
                            ;;
        * )                 break
                            ;;
    esac
done

# check if correct number of arguments are given
if [ "$#" -ne 2 ]; then
    usage
fi

SRC_DIR="$1"
TARGET_DIR="$2"

# use rsync to copy the files
if [ $VERBOSE -eq 1 ]; then
    rsync -av -e ssh "$SRC_DIR"/ "$TARGET_DIR"
else
    rsync -a -e ssh "$SRC_DIR"/ "$TARGET_DIR"
fi

# success message
[ $VERBOSE -eq 1 ] && echo "Files copied successfully!"
