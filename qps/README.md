# QR Printing Service
## Target features
- [x] Connects with Dymo LabelWriter ~~550~~ 400 (libusb)
- [x] Converts plaintext to QR code bitmap (libqrencode)
- [x] Acts as a simple TCP server listening on the loopback interface & providing easy way to print a label using RID (reagent identifier), as well as enabling the client to be notified when the printer job state changes. (plibsys socket API)
- [ ] Properly handles double labels

The project is currently aimed at handling the Dymo 11353 labels only (Label size or eppendorfs - 25 x 13 mm)

## Compilation
### Compile on the target system
1. Create a build directory
2. `cd` into the build directory
3. Run `cmake -DWITHOUT_PNG=ON -DCMAKE_USE_PTHREADS_INIT=OFF -DCMAKE_THREAD_PREFER_PTHREAD=OFF ..`
4. Run `cmake --build .`

### Cross-compile for Windows
1. Create a build directory
2. `cd` into the build directory
3. Run `cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/windows-toolchain.cmake -DPLIBSYS_BUILD_STATIC=ON -DWITHOUT_PNG=ON -DCMAKE_USE_PTHREADS_INIT=OFF -DCMAKE_THREAD_PREFER_PTHREAD=OFF -S ..`
4. Run `cmake --build .`

### Enable libusb-cmake debug logging
You can do it by setting option `-DLIBUSB_ENABLE_DEBUG_LOGGING=ON` with cmake
