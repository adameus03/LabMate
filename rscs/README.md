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
## How it works
- Check issue https://github.com/adameus03/LabMate/issues/13 for available control files.
- To initialize a device you can navigate to `/mnt/rscs` and use `MYSID=$(cat uhfd/sid); echo -n $MYSID > uhfd/mkdev; cd "uhf$(cat uhfd/result/$MYSID/value)"`, which will place you in the directory of the newly created device. Next write required values to the `epc`, `access_passwd` and `kill_passwd` files. You can then write a `1` to `./driver/embody`, which will basically transfer the provided data into the nearby physical tag present in the field.
- If the tag was already embodied, then you need to initialize it with this information by additionaly writing `echo -n 02 > ./flags`
- To read latest measurement values, read from the `rssi` and `read_rate` files. In order to trigger a measurement, there are 2 ways: (a) trigger a full measurement by writing a microseconds timeout integer value to `./driver/embody` or (b) write a `-1` to the same file to trigger a quick measurement (rssi only, it tries to access the tag only once) (please use `echo -n`)  
- Full documentation needs to be written yet

## Compilation
### Compile on the target system
1. Create a build directory
2. `cd` into the build directory
3. Run ~`cmake -DCMAKE_USE_PTHREADS_INIT=OFF -DCMAKE_THREAD_PREFER_PTHREAD=OFF ..`~ `cmake -S .. -B .`
4. Run `cmake --build .`

## Runnining
Mount the filesystem
`rscs -f /mnt/rscs`,
Unmount the filsystem
`umount /mnt/rscs`

### ~Enable libusb-cmake debug logging~
~You can do it by setting option `-DLIBUSB_ENABLE_DEBUG_LOGGING=ON` with cmake~

## Generate documentation
To generate the documentation with Doxygen, run `doxygen` in the RSCS project directory.
