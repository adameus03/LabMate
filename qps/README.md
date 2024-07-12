# QR Printing Service
## Target features
- [ ] Connects with Dymo LabelWriter 550 (libusb)
- [ ] Converts plaintext to QR code bitmap (libqrencode)
- [x] Acts as a simple TCP server listening on the loopback interface & providing easy way to print a label using RID (reagent identifier), as well as enabling the client to be notified when the printer job state changes.

## Compilation
### Compile on the target system
1. Create a build directory
2. `cd` into the build directory
3. Run `cmake ..`
4. Run 'cmake --build .`

### Cross-compile for Windows
1. Create a build directory
2. `cd` into the build directory
3. Run `cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/windows-toolchain.cmake -DPLIBSYS_BUILD_STATIC=ON -S ..`
4. Run `cmake --build .`

