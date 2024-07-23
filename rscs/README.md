# Reagent Scanning Control Subsystem
WIP
## Target features
- [ ] Connects with YPD-R200
- [ ] Provides API for controlling RID identifiers used as a base to transmit RTK (Reagent Tag Key) to the YPD-R200 writer electronics
- [ ] Provides a data source for receiving attributes specified in a subscribtion - in realtime & leveraging data obtained from the YPD-R200 reader module

The project is currently aimed at handling the YPD-R200 module only.


## Compilation
### Compile on the target system
1. Create a build directory
2. `cd` into the build directory
3. Run `cmake -DCMAKE_USE_PTHREADS_INIT=OFF -DCMAKE_THREAD_PREFER_PTHREAD=OFF ..`
4. Run `cmake --build .`

### Enable libusb-cmake debug logging
You can do it by setting option `-DLIBUSB_ENABLE_DEBUG_LOGGING=ON` with cmake
