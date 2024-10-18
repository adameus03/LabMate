# Reagent Scanning Control Subsystem
This is a driver implemented in userspace enabling controlling a UHF RFID interrogator (only YPD-R200 for now).
The driver functionality is exposed as a filesystem in userspace (FUSE).
## Target features
- [x] Connects with YPD-R200
- [x] Enables controlling RID identifiers used as a base to transmit RTK (Reagent Tag Key) to the YPD-R200 writer electronics
- [x] Provides a data source for receiving tag measurements such as RSSI and read rate

The project is currently aimed at handling the YPD-R200 module only, but a HAL-like layer exists and is called `uhfman`.
The structure of the project is as follows:
```
bash/client software --- database <<< external to RSCS itself
         |
    (filesystem interface)
         |
       main (FUSE)
         |
       uhfd (UHF RFID tag devs driver)
        |
      uhfman (abstracted management for interrogator device / HAL)
        |
     ypdr200 (interrogator-specific device driver)
```

## Compilation
### Compile on the target system
1. Create a build directory
2. `cd` into the build directory
3. Run ~`cmake -DCMAKE_USE_PTHREADS_INIT=OFF -DCMAKE_THREAD_PREFER_PTHREAD=OFF ..`~ `cmake -S .. -B .`
4. Run `cmake --build .`

## Runnining
Mount the filesystem
`rscs -f /mnt/rscs`
Unmount the filsystem
`umount /mnt/rscs`

### ~Enable libusb-cmake debug logging~
~You can do it by setting option `-DLIBUSB_ENABLE_DEBUG_LOGGING=ON` with cmake~

## Generate documentation
To generate the documentation with Doxygen, run `doxygen` in the RSCS project directory.
