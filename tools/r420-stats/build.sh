#!/bin/bash
#set -e
#set -o pipefail
set -u
#set -x

cd "$(dirname "$0")"
if [ $? -ne 0 ]; then
  echo "build: Failed to change directory to build script location"
  exit 1
fi
as --64 main.s -o main.o && as --64 util.s -o util.o
if [ $? -ne 0 ]; then
  echo "build: Assembly failed"
  exit 1
fi
echo "build: Assembly completed successfully"
ld ./*.o -o test
if [ $? -ne 0 ]; then
  echo "build: Linking failed"
  exit 1
fi
rm ./*.o
if [ $? -ne 0 ]; then
  echo "build: Failed to remove intermediate object files"
  exit 1
fi
echo "build: Build completed successfully"
chmod +x test
if [ $? -ne 0 ]; then
  echo "build: Failed to set executable permissions"
  exit 1
fi
echo "build: Executable is ready: $(pwd)/test"
if [ $? -ne 0 ]; then
  echo "build: Failed to get current directory"
  exit 1
fi
exit 0