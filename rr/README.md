## LabMate-RR (Redis Receiver) - FIFO mq consumers for labserv
###
### Building (GNU/Linux)
You need to have dependency packages installed. Here is the dependency tree produced by `lddtree rr` run on a Debian stable release (bookworm):
```shell
$ lddtree rr
rr (interpreter => /lib64/ld-linux-x86-64.so.2)
    libhiredis.so.0.14 => /lib/x86_64-linux-gnu/libhiredis.so.0.14
    libplibsys.so.0 => /usr/local/lib/libplibsys.so.0
    libcurl.so.4 => /lib/x86_64-linux-gnu/libcurl.so.4
        libnghttp2.so.14 => /lib/x86_64-linux-gnu/libnghttp2.so.14
        libidn2.so.0 => /lib/x86_64-linux-gnu/libidn2.so.0
            libunistring.so.2 => /lib/x86_64-linux-gnu/libunistring.so.2
        librtmp.so.1 => /lib/x86_64-linux-gnu/librtmp.so.1
            libgnutls.so.30 => /lib/x86_64-linux-gnu/libgnutls.so.30
                libp11-kit.so.0 => /lib/x86_64-linux-gnu/libp11-kit.so.0
                    libffi.so.8 => /lib/x86_64-linux-gnu/libffi.so.8
                libtasn1.so.6 => /lib/x86_64-linux-gnu/libtasn1.so.6
            libhogweed.so.6 => /lib/x86_64-linux-gnu/libhogweed.so.6
            libnettle.so.8 => /lib/x86_64-linux-gnu/libnettle.so.8
            libgmp.so.10 => /lib/x86_64-linux-gnu/libgmp.so.10
        libssh2.so.1 => /lib/x86_64-linux-gnu/libssh2.so.1
        libpsl.so.5 => /lib/x86_64-linux-gnu/libpsl.so.5
        libssl.so.3 => /lib/x86_64-linux-gnu/libssl.so.3
        libcrypto.so.3 => /lib/x86_64-linux-gnu/libcrypto.so.3
        libgssapi_krb5.so.2 => /lib/x86_64-linux-gnu/libgssapi_krb5.so.2
            libkrb5.so.3 => /lib/x86_64-linux-gnu/libkrb5.so.3
                libkeyutils.so.1 => /lib/x86_64-linux-gnu/libkeyutils.so.1
                libresolv.so.2 => /lib/x86_64-linux-gnu/libresolv.so.2
            libk5crypto.so.3 => /lib/x86_64-linux-gnu/libk5crypto.so.3
            libcom_err.so.2 => /lib/x86_64-linux-gnu/libcom_err.so.2
            libkrb5support.so.0 => /lib/x86_64-linux-gnu/libkrb5support.so.0
        libldap-2.5.so.0 => /lib/x86_64-linux-gnu/libldap-2.5.so.0
            libsasl2.so.2 => /lib/x86_64-linux-gnu/libsasl2.so.2
        liblber-2.5.so.0 => /lib/x86_64-linux-gnu/liblber-2.5.so.0
        libzstd.so.1 => /lib/x86_64-linux-gnu/libzstd.so.1
        libbrotlidec.so.1 => /lib/x86_64-linux-gnu/libbrotlidec.so.1
            libbrotlicommon.so.1 => /lib/x86_64-linux-gnu/libbrotlicommon.so.1
        libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
```

This means that having hiredis, plibsys and curl development packages should be enough to build and run rr. Depending on your distribution, you may need to compile and install plibsys from source (see Containerfile for hint). As for curl and hiredis you can usually install them from your package manager. For example, on Debian stable release (bookworm) you can install them with:
```shell
$ sudo apt install libcurl4-openssl-dev libhiredis-dev
``` 
It is assumed that you have the gcc compiler, make and cmake installed. If not, you can install them with:
```shell
$ sudo apt install build-essential cmake
```
To configure and build rr, you can use the following commands:
```shell
$ mkdir build
$ cd build
$ cmake -S .. -B .
$ make
```
It is possible to configure cmake options with `-D<option>=<value>` syntax. To enable address and leak sanitizer for example, you can use:
```shell
$ cmake -DFSANITIZE_LA=ON -S .. -B .
$ make
```
For more options, you can check CMakeLists.txt file in the root directory of rr.
###
Note: beware that you need to have a Redis server running so that rr can `BLPOP` from its message queues. You thus need to setup the constants in `config.h` as needed. You can use `config-sample.h` as a template for configuration.

###
### If you want to run as container (podman) - for quick deployment and development purposes on various OS platforms
To run the below commands you need to have podman installed (or an OCI compatible alternative like docker). You also need to be in the root directory of rr.
To build the container image, you can use the following command:
```shell
$ podman --runtime crun --storage-driver overlay --root /tmp/podman-storage --runroot /tmp/podman-runroot build . -t labmate-rr
```
To run the container:
```shell
$ podman --runtime crun --storage-driver overlay --root /tmp/podman-storage --runroot /tmp/podman-runroot run --name LABMATE-RR --net=host -it labmate-rr
```
To stop the container instantly:
```shell
podman --runtime crun --storage-driver overlay --root /tmp/podman-storage --runroot /tmp/podman-runroot stop LABMATE-RR -t0
```
To remove the container:
```shell
podman --runtime crun --storage-driver overlay --root /tmp/podman-storage --runroot /tmp/podman-runroot rm LABMATE-RR
```
Note: using the overlay filesystem is highly advantegous, as it saves us from wasting disk space (differences in comparison to VFS are like <1GB vs 35GB for a single container !!)

