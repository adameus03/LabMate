#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <math.h>
//#include <drm/drm_mode.h>

int main(void) {
    int fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (fd < 0) { perror("open"); return 1; }

    drmModeRes *res = drmModeGetResources(fd);
    if (!res) { fprintf(stderr, "drmModeGetResources failed\n"); return 1; }

    drmModeConnector *conn = NULL;
    for (int i = 0; i < res->count_connectors; i++) {
        conn = drmModeGetConnector(fd, res->connectors[i]);
        if (conn->connection == DRM_MODE_CONNECTED) break;
        drmModeFreeConnector(conn);
        conn = NULL;
    }
    if (!conn) { fprintf(stderr, "No connected connector\n"); return 1; }

    drmModeModeInfo mode = conn->modes[0]; // choose first mode
    drmModeEncoder *enc = drmModeGetEncoder(fd, conn->encoder_id);

    struct drm_mode_create_dumb creq = {0};
    creq.width  = mode.hdisplay;
    creq.height = mode.vdisplay;
    creq.bpp    = 32; // 4 bytes per pixel
    if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &creq) < 0) {
        perror("DRM_IOCTL_MODE_CREATE_DUMB"); return 1;
    }

    // Map framebuffer
    struct drm_mode_map_dumb mreq = {0};
    mreq.handle = creq.handle;
    if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &mreq) < 0) {
        perror("DRM_IOCTL_MODE_MAP_DUMB"); return 1;
    }

    void *fb = mmap(0, creq.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mreq.offset);
    if (fb == MAP_FAILED) { perror("mmap"); return 1; }

    // Fill framebuffer with red pixels
    uint32_t *pixels = fb;
    for (int y = 0; y < mode.vdisplay; y++) {
        for (int x = 0; x < mode.hdisplay; x++) {
            pixels[y*mode.hdisplay + x] = 0x00FF0000; // ARGB
            //pixels[y*mode.hdisplay + x] = 1000*x+10000*y+5000*cos(x/20.0)*cos(y/20.0);
        }
    }

    // Create framebuffer object
    uint32_t fb_id;
    if (drmModeAddFB(fd, mode.hdisplay, mode.vdisplay, 24, 32, creq.pitch, creq.handle, &fb_id)) {
        perror("drmModeAddFB"); return 1;
    }

    // Set CRTC to display our buffer
    if (drmModeSetCrtc(fd, enc->crtc_id, fb_id, 0, 0, &conn->connector_id, 1, &mode)) {
        perror("drmModeSetCrtc"); return 1;
    }

    printf("Press Enter to exit...\n");
    getchar();

    // Cleanup (simplified)
    drmModeRmFB(fd, fb_id);
    munmap(fb, creq.size);

    close(fd);
    drmModeFreeEncoder(enc);
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);

    return 0;
}
