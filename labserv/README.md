## labserv (LABmate cloud SERVer)
"Cloud" here means that the server doesn't necessarily need to be run on an on-premises equipment. It can be run on a cloud provider's infrastructure. The server is designed to be a central point of communication between clients and the on-premises part of LabMate. It is a RESTful API server that maintains a pool of postgreSQL database connections and provides a set of endpoints for clients to interact with the database. The server is written in C and uses libh2o for HTTP server functionality and libpq for database connectivity. The server is designed to be run as a daemon and can be configured with a configuration header file (`config.h`).
The server is developed and tested on GNU/Linux systems (Debian stable release - bookworm and Alpine Linux).
###
### Building (GNU/Linux)
You need to have dependency packages installed. Here is the dependency tree produced by `lddtree labserv` run on a Debian stable release (bookworm):
```shell
$ lddtree labserv
labserv (interpreter => /lib64/ld-linux-x86-64.so.2)
    libuv.so.1 => /lib/x86_64-linux-gnu/libuv.so.1
    libssl.so.3 => /lib/x86_64-linux-gnu/libssl.so.3
    libcrypto.so.3 => /lib/x86_64-linux-gnu/libcrypto.so.3
    libh2o.so.0.13 => /lib/x86_64-linux-gnu/libh2o.so.0.13
        libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1
        libwslay.so.1 => /lib/x86_64-linux-gnu/libwslay.so.1
    libpq.so.5 => /lib/x86_64-linux-gnu/libpq.so.5
        libgssapi_krb5.so.2 => /lib/x86_64-linux-gnu/libgssapi_krb5.so.2
            libkrb5.so.3 => /lib/x86_64-linux-gnu/libkrb5.so.3
                libkeyutils.so.1 => /lib/x86_64-linux-gnu/libkeyutils.so.1
                libresolv.so.2 => /lib/x86_64-linux-gnu/libresolv.so.2
            libk5crypto.so.3 => /lib/x86_64-linux-gnu/libk5crypto.so.3
            libcom_err.so.2 => /lib/x86_64-linux-gnu/libcom_err.so.2
            libkrb5support.so.0 => /lib/x86_64-linux-gnu/libkrb5support.so.0
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6
        libldap-2.5.so.0 => /lib/x86_64-linux-gnu/libldap-2.5.so.0
            liblber-2.5.so.0 => /lib/x86_64-linux-gnu/liblber-2.5.so.0
            libsasl2.so.2 => /lib/x86_64-linux-gnu/libsasl2.so.2
            libgnutls.so.30 => /lib/x86_64-linux-gnu/libgnutls.so.30
                libp11-kit.so.0 => /lib/x86_64-linux-gnu/libp11-kit.so.0
                    libffi.so.8 => /lib/x86_64-linux-gnu/libffi.so.8
                libidn2.so.0 => /lib/x86_64-linux-gnu/libidn2.so.0
                libunistring.so.2 => /lib/x86_64-linux-gnu/libunistring.so.2
                libtasn1.so.6 => /lib/x86_64-linux-gnu/libtasn1.so.6
                libnettle.so.8 => /lib/x86_64-linux-gnu/libnettle.so.8
                libhogweed.so.6 => /lib/x86_64-linux-gnu/libhogweed.so.6
                libgmp.so.10 => /lib/x86_64-linux-gnu/libgmp.so.10
    libbcrypt.so.1 => /usr/local/lib/libbcrypt.so.1
    libplibsys.so.0 => /usr/local/lib/libplibsys.so.0
    libhiredis.so.0.14 => /lib/x86_64-linux-gnu/libhiredis.so.0.14
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
```

This means that having libuv, libssl, libcrypto, libh2o, libpq, libbcrypt, libplibsys and libhiredis development packages should be enough to build and run labserv.
Depending on your distribution, you may need to compile and install plibsys, h2o and libbcrypt from source (see Containerfile for hints). As for other packages you can usually install them from your package manager. For example, on Debian stable release (bookworm) you can install them with:
```shell
$ sudo apt install libuv1-dev libssl3 libpq5 libhiredis 
``` 
It is assumed that you have the gcc compiler, make and cmake installed. If not, you can install them with:
```shell
$ sudo apt install build-essential cmake
```
To configure and build labserv, you can use the following commands:
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
Note: beware that you need to have a postgreSQL server running and a database created for labserv to connect to. Same applies to Redis. You need to setup the constants in `config.h` as needed. You can use `config-sample.h` as a template for configuration.

###
### If you want to run as container (podman) - for quick deployment and development purposes on various OS platforms
To run the below commands you need to have podman installed (or an OCI compatible alternative like docker). You also need to be in the root directory of rr.
To build the container image, you can use the following command:
```shell
$ podman --runtime crun --storage-driver overlay --root /tmp/podman-storage --runroot /tmp/podman-runroot build . -t labserv
```
To run the container:
```shell
$ podman --runtime crun --storage-driver overlay --root /tmp/podman-storage --runroot /tmp/podman-runroot run --name LABSERV --net=host -it labserv
```
To stop the container instantly:
```shell
podman --runtime crun --storage-driver overlay --root /tmp/podman-storage --runroot /tmp/podman-runroot stop LABSERV -t0
```
To remove the container:
```shell
podman --runtime crun --storage-driver overlay --root /tmp/podman-storage --runroot /tmp/podman-runroot rm LABSERV
```
Note: using the overlay filesystem is highly advantegous, as it saves us from wasting disk space (differences in comparison to VFS are like <1GB vs 35GB for a single container !!)

