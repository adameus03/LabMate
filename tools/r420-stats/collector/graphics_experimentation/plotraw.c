/*
 * DRM/KMS framebuffer with three keyboards controlling three circles
 * Circles move continuously while keys are held down
 * Compile: gcc drm_safe.c -ldrm -o drm_safe
 * Run: switch to a spare VT (Ctrl+Alt+F2), then sudo ./drm_safe
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <signal.h>
#include <linux/input.h>
#include <dirent.h>

#include <xf86drm.h>
#include <xf86drmMode.h>

#include <math.h>

int fd = -1;
drmModeCrtc *old_crtc = NULL;
drmModeConnector *conn = NULL;
drmModeEncoder *enc = NULL;
uint32_t fb_id = 0;
void *fb_map = NULL;
size_t fb_size = 0;
drmModeModeInfo mode;

int center1_x, center1_y;  // Red circle (keyboard 1)
int center2_x, center2_y;  // Green circle (keyboard 2)
int center3_x, center3_y;  // Yellow circle (keyboard 3)

int radius1 = 50;  // Red circle radius
int radius2 = 50;  // Green circle radius
int radius3 = 50;  // Yellow circle radius

int charge1 = 0;  // Red circle charge
int charge2 = 0;  // Green circle charge
int charge3 = 0;  // Yellow circle charge

int vel1_x = 0, vel1_y = 0;  // Red circle velocity
int vel2_x = 0, vel2_y = 0;  // Green circle velocity
int vel3_x = 0, vel3_y = 0;  // Yellow circle velocity

int kbd1_fd = -1;  // Keyboard 1
int kbd2_fd = -1;  // Keyboard 2
int kbd3_fd = -1;  // Keyboard 3

// Key states for keyboard 1
int kbd1_w = 0, kbd1_s = 0, kbd1_a = 0, kbd1_d = 0;
int kbd1_up = 0, kbd1_down = 0, kbd1_left = 0, kbd1_right = 0;
int kbd1_p = 0, kbd1_o = 0, kbd1_r = 0, kbd1_e = 0;

// Key states for keyboard 2
int kbd2_w = 0, kbd2_s = 0, kbd2_a = 0, kbd2_d = 0;
int kbd2_up = 0, kbd2_down = 0, kbd2_left = 0, kbd2_right = 0;
int kbd2_p = 0, kbd2_o = 0, kbd2_r = 0, kbd2_e = 0;

// Key states for keyboard 3
int kbd3_w = 0, kbd3_s = 0, kbd3_a = 0, kbd3_d = 0;
int kbd3_up = 0, kbd3_down = 0, kbd3_left = 0, kbd3_right = 0;
int kbd3_p = 0, kbd3_o = 0, kbd3_r = 0, kbd3_e = 0;

int should_exit = 0;

void cleanup(void) {
    if (kbd1_fd >= 0) { close(kbd1_fd); kbd1_fd = -1; }
    if (kbd2_fd >= 0) { close(kbd2_fd); kbd2_fd = -1; }
    if (kbd3_fd >= 0) { close(kbd3_fd); kbd3_fd = -1; }
    
    if (fb_map) {
        munmap(fb_map, fb_size);
        fb_map = NULL;
    }

    if (fb_id && enc && conn && fd >= 0) {
        if (old_crtc)
            drmModeSetCrtc(fd, old_crtc->crtc_id, old_crtc->buffer_id,
                           old_crtc->x, old_crtc->y, &conn->connector_id, 1,
                           &old_crtc->mode);
        drmModeRmFB(fd, fb_id);
        fb_id = 0;
    }

    if (enc) { drmModeFreeEncoder(enc); enc = NULL; }
    if (conn) { drmModeFreeConnector(conn); conn = NULL; }
    if (old_crtc) { drmModeFreeCrtc(old_crtc); old_crtc = NULL; }
    if (fd >= 0) { close(fd); fd = -1; }
}

void vt_handler(int sig) {
    cleanup();
    exit(0);
}

void update_framebuffer(void) {
    uint32_t *pixels = fb_map;
    for (int y = 0; y < mode.vdisplay; y++) {
        for (int x = 0; x < mode.hdisplay; x++) {
            int dist1 = (x-center1_x)*(x-center1_x) + (y-center1_y)*(y-center1_y);
            int dist2 = (x-center2_x)*(x-center2_x) + (y-center2_y)*(y-center2_y);
            int dist3 = (x-center3_x)*(x-center3_x) + (y-center3_y)*(y-center3_y);
            
            if (dist1 < radius1*radius1) {
                // Red circle - color based on charge
                if (charge1 > 0) {
                    // Positive charge: more red
                    int intensity = (charge1 > 100) ? 255 : (charge1 * 255 / 100);
                    pixels[y*mode.hdisplay + x] = 0x00FF0000 | (intensity << 16);
                } else if (charge1 < 0) {
                    // Negative charge: add blue tint
                    int blue = (-charge1 > 100) ? 255 : (-charge1 * 255 / 100);
                    pixels[y*mode.hdisplay + x] = 0x00FF0000 | blue;
                } else {
                    // Neutral
                    pixels[y*mode.hdisplay + x] = 0x00FF0000;
                }
            } else if (dist2 < radius2*radius2) {
                // Green circle - color based on charge
                if (charge2 > 0) {
                    // Positive charge: add red tint
                    int red = (charge2 > 100) ? 255 : (charge2 * 255 / 100);
                    pixels[y*mode.hdisplay + x] = 0x0000FF00 | (red << 16);
                } else if (charge2 < 0) {
                    // Negative charge: add blue tint
                    int blue = (-charge2 > 100) ? 255 : (-charge2 * 255 / 100);
                    pixels[y*mode.hdisplay + x] = 0x0000FF00 | blue;
                } else {
                    // Neutral
                    pixels[y*mode.hdisplay + x] = 0x0000FF00;
                }
            } else if (dist3 < radius3*radius3) {
                // Yellow circle - color based on charge
                if (charge3 > 0) {
                    // Positive charge: brighter yellow/orange
                    pixels[y*mode.hdisplay + x] = 0x00FFFF00;
                } else if (charge3 < 0) {
                    // Negative charge: add blue tint (greenish)
                    int blue = (-charge3 > 100) ? 200 : (-charge3 * 200 / 100);
                    pixels[y*mode.hdisplay + x] = 0x00FFFF00 | blue;
                } else {
                    // Neutral
                    pixels[y*mode.hdisplay + x] = 0x00FFFF00;
                }
            } else {
                // Blue background
                pixels[y*mode.hdisplay + x] = 0x000000FF;
            }
        }
    }
    
    drmModeDirtyFB(fd, fb_id, NULL, 0);
}

void update_key_state(struct input_event *ev, int *w, int *s, int *a, int *d,
                      int *up, int *down, int *left, int *right, int *p, int *o,
                      int *r, int *e) {
    if (ev->type != EV_KEY) return;
    
    int pressed = (ev->value == 1 || ev->value == 2);  // 1=press, 2=repeat, 0=release
    
    switch(ev->code) {
        case KEY_W:     *w = pressed; break;
        case KEY_S:     *s = pressed; break;
        case KEY_A:     *a = pressed; break;
        case KEY_D:     *d = pressed; break;
        case KEY_UP:    *up = pressed; break;
        case KEY_DOWN:  *down = pressed; break;
        case KEY_LEFT:  *left = pressed; break;
        case KEY_RIGHT: *right = pressed; break;
        case KEY_P:     *p = pressed; break;
        case KEY_O:     *o = pressed; break;
        case KEY_R:     *r = pressed; break;
        case KEY_E:     *e = pressed; break;
        case KEY_Q:
        case KEY_ESC:
            if (pressed) should_exit = 1;
            break;
    }
}

void handle_input(void) {
    struct input_event ev;
    
    // Read from keyboard 1
    if (kbd1_fd >= 0) {
        while (read(kbd1_fd, &ev, sizeof(ev)) == sizeof(ev)) {
            update_key_state(&ev, &kbd1_w, &kbd1_s, &kbd1_a, &kbd1_d,
                           &kbd1_up, &kbd1_down, &kbd1_left, &kbd1_right,
                           &kbd1_p, &kbd1_o, &kbd1_r, &kbd1_e);
        }
    }
    
    // Read from keyboard 2
    if (kbd2_fd >= 0) {
        while (read(kbd2_fd, &ev, sizeof(ev)) == sizeof(ev)) {
            update_key_state(&ev, &kbd2_w, &kbd2_s, &kbd2_a, &kbd2_d,
                           &kbd2_up, &kbd2_down, &kbd2_left, &kbd2_right,
                           &kbd2_p, &kbd2_o, &kbd2_r, &kbd2_e);
        }
    }
    
    // Read from keyboard 3
    if (kbd3_fd >= 0) {
        while (read(kbd3_fd, &ev, sizeof(ev)) == sizeof(ev)) {
            update_key_state(&ev, &kbd3_w, &kbd3_s, &kbd3_a, &kbd3_d,
                           &kbd3_up, &kbd3_down, &kbd3_left, &kbd3_right,
                           &kbd3_p, &kbd3_o, &kbd3_r, &kbd3_e);
        }
    }
}

void update_positions(void) {
    int speed = 5;  // Pixels per frame
    int grow_speed = 2;  // Radius change per frame
    int charge_speed = 2;  // Charge change per frame
    
    // Store desired velocities based on key states
    vel1_x = vel1_y = 0;
    vel2_x = vel2_y = 0;
    vel3_x = vel3_y = 0;
    
    // Keyboard 1 velocity
    if (kbd1_w || kbd1_up) vel1_y -= speed;
    if (kbd1_s || kbd1_down) vel1_y += speed;
    if (kbd1_a || kbd1_left) vel1_x -= speed;
    if (kbd1_d || kbd1_right) vel1_x += speed;
    
    // Keyboard 2 velocity
    if (kbd2_w || kbd2_up) vel2_y -= speed;
    if (kbd2_s || kbd2_down) vel2_y += speed;
    if (kbd2_a || kbd2_left) vel2_x -= speed;
    if (kbd2_d || kbd2_right) vel2_x += speed;
    
    // Keyboard 3 velocity
    if (kbd3_w || kbd3_up) vel3_y -= speed;
    if (kbd3_s || kbd3_down) vel3_y += speed;
    if (kbd3_a || kbd3_left) vel3_x -= speed;
    if (kbd3_d || kbd3_right) vel3_x += speed;
    
    // Apply electrostatic forces between circles
    //float k = 0.5;  // Coulomb constant (scaled for visual effect)
    float k = 100;
    
    // Force between circles 1 and 2
    int dx12 = center2_x - center1_x;
    int dy12 = center2_y - center1_y;
    float dist12_sq = dx12*dx12 + dy12*dy12;
    if (dist12_sq > 100) {  // Avoid division by very small numbers
        float dist12 = sqrtf(dist12_sq);
        float force12 = k * charge1 * charge2 / dist12_sq;
        float fx12 = force12 * dx12 / dist12;
        float fy12 = force12 * dy12 / dist12;
        vel1_x += (int)fx12;
        vel1_y += (int)fy12;
        vel2_x -= (int)fx12;
        vel2_y -= (int)fy12;
    }
    
    // Force between circles 1 and 3
    int dx13 = center3_x - center1_x;
    int dy13 = center3_y - center1_y;
    float dist13_sq = dx13*dx13 + dy13*dy13;
    if (dist13_sq > 100) {
        float dist13 = sqrtf(dist13_sq);
        float force13 = k * charge1 * charge3 / dist13_sq;
        float fx13 = force13 * dx13 / dist13;
        float fy13 = force13 * dy13 / dist13;
        vel1_x += (int)fx13;
        vel1_y += (int)fy13;
        vel3_x -= (int)fx13;
        vel3_y -= (int)fy13;
    }
    
    // Force between circles 2 and 3
    int dx23 = center3_x - center2_x;
    int dy23 = center3_y - center2_y;
    float dist23_sq = dx23*dx23 + dy23*dy23;
    if (dist23_sq > 100) {
        float dist23 = sqrtf(dist23_sq);
        float force23 = k * charge2 * charge3 / dist23_sq;
        float fx23 = force23 * dx23 / dist23;
        float fy23 = force23 * dy23 / dist23;
        vel2_x += (int)fx23;
        vel2_y += (int)fy23;
        vel3_x -= (int)fx23;
        vel3_y -= (int)fy23;
    }
    
    // Apply velocities
    center1_x += vel1_x;
    center1_y += vel1_y;
    center2_x += vel2_x;
    center2_y += vel2_y;
    center3_x += vel3_x;
    center3_y += vel3_y;
    
    // Check collision between circles 1 and 2 (still maintain physical collision)
    dx12 = center2_x - center1_x;
    dy12 = center2_y - center1_y;
    dist12_sq = dx12*dx12 + dy12*dy12;
    int min_dist12 = radius1 + radius2;
    
    if (dist12_sq < min_dist12*min_dist12 && dist12_sq > 0) {
        // Collision detected - push circles apart
        float dist12 = sqrtf(dist12_sq);
        float overlap = min_dist12 - dist12;
        float dx_norm = dx12 / dist12;
        float dy_norm = dy12 / dist12;
        
        // Push each circle back by half the overlap
        center1_x -= (int)(dx_norm * overlap / 2);
        center1_y -= (int)(dy_norm * overlap / 2);
        center2_x += (int)(dx_norm * overlap / 2);
        center2_y += (int)(dy_norm * overlap / 2);
    }
    
    // Check collision between circles 1 and 3
    int dx13_new = center3_x - center1_x;
    int dy13_new = center3_y - center1_y;
    float dist13_sq_new = dx13_new*dx13_new + dy13_new*dy13_new;
    int min_dist13 = radius1 + radius3;
    
    if (dist13_sq_new < min_dist13*min_dist13 && dist13_sq_new > 0) {
        float dist13 = sqrtf(dist13_sq_new);
        float overlap = min_dist13 - dist13;
        float dx_norm = dx13_new / dist13;
        float dy_norm = dy13_new / dist13;
        
        center1_x -= (int)(dx_norm * overlap / 2);
        center1_y -= (int)(dy_norm * overlap / 2);
        center3_x += (int)(dx_norm * overlap / 2);
        center3_y += (int)(dy_norm * overlap / 2);
    }
    
    // Check collision between circles 2 and 3
    int dx23_new = center3_x - center2_x;
    int dy23_new = center3_y - center2_y;
    float dist23_sq_new = dx23_new*dx23_new + dy23_new*dy23_new;
    int min_dist23 = radius2 + radius3;
    
    if (dist23_sq_new < min_dist23*min_dist23 && dist23_sq_new > 0) {
        float dist23 = sqrtf(dist23_sq_new);
        float overlap = min_dist23 - dist23;
        float dx_norm = dx23_new / dist23;
        float dy_norm = dy23_new / dist23;
        
        center2_x -= (int)(dx_norm * overlap / 2);
        center2_y -= (int)(dy_norm * overlap / 2);
        center3_x += (int)(dx_norm * overlap / 2);
        center3_y += (int)(dy_norm * overlap / 2);
    }
    
    // Boundary checks for circle 1
    if (center1_x < radius1) center1_x = radius1;
    if (center1_x > mode.hdisplay - radius1) center1_x = mode.hdisplay - radius1;
    if (center1_y < radius1) center1_y = radius1;
    if (center1_y > mode.vdisplay - radius1) center1_y = mode.vdisplay - radius1;
    
    // Boundary checks for circle 2
    if (center2_x < radius2) center2_x = radius2;
    if (center2_x > mode.hdisplay - radius2) center2_x = mode.hdisplay - radius2;
    if (center2_y < radius2) center2_y = radius2;
    if (center2_y > mode.vdisplay - radius2) center2_y = mode.vdisplay - radius2;
    
    // Boundary checks for circle 3
    if (center3_x < radius3) center3_x = radius3;
    if (center3_x > mode.hdisplay - radius3) center3_x = mode.hdisplay - radius3;
    if (center3_y < radius3) center3_y = radius3;
    if (center3_y > mode.vdisplay - radius3) center3_y = mode.vdisplay - radius3;
    
    // Handle size changes for circle 1
    if (kbd1_p) {
        radius1 += grow_speed;
        if (radius1 > 200) radius1 = 200;  // Max radius
    }
    if (kbd1_o) {
        radius1 -= grow_speed;
        if (radius1 < 10) radius1 = 10;  // Min radius
    }
    
    // Handle charge changes for circle 1
    if (kbd1_r) {
        charge1 += charge_speed;
        if (charge1 > 100) charge1 = 100;  // Max charge
    }
    if (kbd1_e) {
        charge1 -= charge_speed;
        if (charge1 < -100) charge1 = -100;  // Min charge
    }
    
    // Handle size changes for circle 2
    if (kbd2_p) {
        radius2 += grow_speed;
        if (radius2 > 200) radius2 = 200;
    }
    if (kbd2_o) {
        radius2 -= grow_speed;
        if (radius2 < 10) radius2 = 10;
    }
    
    // Handle charge changes for circle 2
    if (kbd2_r) {
        charge2 += charge_speed;
        if (charge2 > 100) charge2 = 100;
    }
    if (kbd2_e) {
        charge2 -= charge_speed;
        if (charge2 < -100) charge2 = -100;
    }
    
    // Handle size changes for circle 3
    if (kbd3_p) {
        radius3 += grow_speed;
        if (radius3 > 200) radius3 = 200;
    }
    if (kbd3_o) {
        radius3 -= grow_speed;
        if (radius3 < 10) radius3 = 10;
    }
    
    // Handle charge changes for circle 3
    if (kbd3_r) {
        charge3 += charge_speed;
        if (charge3 > 100) charge3 = 100;
    }
    if (kbd3_e) {
        charge3 -= charge_speed;
        if (charge3 < -100) charge3 = -100;
    }
}

int is_keyboard(const char *device_path) {
    int fd = open(device_path, O_RDONLY | O_NONBLOCK);
    if (fd < 0) return 0;
    
    unsigned long evbit = 0;
    ioctl(fd, EVIOCGBIT(0, sizeof(evbit)), &evbit);
    
    // Check if device supports key events
    int is_kbd = (evbit & (1 << EV_KEY)) != 0;
    
    // Get device name
    char name[256] = "Unknown";
    ioctl(fd, EVIOCGNAME(sizeof(name)), name);
    
    close(fd);
    
    // Skip devices that are clearly not keyboards
    if (strstr(name, "Mouse") || strstr(name, "mouse") ||
        strstr(name, "Touchpad") || strstr(name, "touchpad") ||
        strstr(name, "Power") || strstr(name, "power") ||
        strstr(name, "Video") || strstr(name, "Sleep") ||
        strstr(name, "Lid")) {
        return 0;
    }
    
    return is_kbd;
}

void find_keyboards(void) {
    DIR *dir = opendir("/dev/input");
    if (!dir) {
        perror("opendir /dev/input");
        return;
    }
    
    printf("\nAvailable keyboards:\n");
    printf("====================\n");
    
    struct dirent *entry;
    int kbd_count = 0;
    
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "event", 5) != 0) continue;
        
        char path[256];
        snprintf(path, sizeof(path), "/dev/input/%s", entry->d_name);
        
        if (!is_keyboard(path)) continue;
        
        int fd = open(path, O_RDONLY);
        if (fd < 0) continue;
        
        char name[256] = "Unknown";
        ioctl(fd, EVIOCGNAME(sizeof(name)), name);
        
        printf("[%d] %s\n    Device: %s\n", kbd_count, path, name);
        
        close(fd);
        kbd_count++;
    }
    
    closedir(dir);
    
    if (kbd_count == 0) {
        printf("No keyboards found!\n");
    }
    printf("\n");
}

int open_keyboard_device(const char *path) {
    int fd = open(path, O_RDONLY | O_NONBLOCK);
    if (fd < 0) {
        perror("open keyboard");
        return -1;
    }
    
    // Grab exclusive access to prevent input from reaching the terminal
    if (ioctl(fd, EVIOCGRAB, 1) < 0) {
        fprintf(stderr, "Warning: Could not grab exclusive access to %s\n", path);
    }
    
    char name[256] = "Unknown";
    ioctl(fd, EVIOCGNAME(sizeof(name)), name);
    printf("Opened: %s (%s)\n", path, name);
    
    return fd;
}

int main(void) {
    signal(SIGINT, vt_handler);
    signal(SIGTERM, vt_handler);
    signal(SIGUSR1, vt_handler);

    // Find and display available keyboards
    find_keyboards();
    
    // Open keyboard devices
    printf("Enter path for keyboard 1 (red circle, e.g., /dev/input/event3): ");
    char kbd1_path[256];
    if (scanf("%255s", kbd1_path) != 1) {
        fprintf(stderr, "Failed to read keyboard 1 path\n");
        return 1;
    }
    
    printf("Enter path for keyboard 2 (green circle, e.g., /dev/input/event4): ");
    char kbd2_path[256];
    if (scanf("%255s", kbd2_path) != 1) {
        fprintf(stderr, "Failed to read keyboard 2 path\n");
        return 1;
    }
    
    printf("Enter path for keyboard 3 (yellow circle, e.g., /dev/input/event5): ");
    char kbd3_path[256];
    if (scanf("%255s", kbd3_path) != 1) {
        fprintf(stderr, "Failed to read keyboard 3 path\n");
        return 1;
    }
    
    kbd1_fd = open_keyboard_device(kbd1_path);
    kbd2_fd = open_keyboard_device(kbd2_path);
    kbd3_fd = open_keyboard_device(kbd3_path);
    
    if (kbd1_fd < 0 || kbd2_fd < 0 || kbd3_fd < 0) {
        fprintf(stderr, "Failed to open keyboard devices\n");
        cleanup();
        return 1;
    }

    fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (fd < 0) { perror("open"); return 1; }

    drmModeRes *res = drmModeGetResources(fd);
    if (!res) { fprintf(stderr, "drmModeGetResources failed\n"); return 1; }

    for (int i = 0; i < res->count_connectors; i++) {
        conn = drmModeGetConnector(fd, res->connectors[i]);
        if (conn->connection == DRM_MODE_CONNECTED) break;
        drmModeFreeConnector(conn);
        conn = NULL;
    }
    if (!conn) { fprintf(stderr, "No connected connector\n"); return 1; }

    enc = drmModeGetEncoder(fd, conn->encoder_id);
    if (!enc) { fprintf(stderr, "No encoder\n"); return 1; }

    old_crtc = drmModeGetCrtc(fd, enc->crtc_id);
    mode = conn->modes[0];

    // Initialize circle centers
    center1_x = mode.hdisplay / 4;
    center1_y = mode.vdisplay / 2;
    center2_x = mode.hdisplay / 2;
    center2_y = mode.vdisplay / 2;
    center3_x = 3 * mode.hdisplay / 4;
    center3_y = mode.vdisplay / 2;

    struct drm_mode_create_dumb creq = {0};
    creq.width  = mode.hdisplay;
    creq.height = mode.vdisplay;
    creq.bpp    = 32;
    if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &creq) < 0) {
        perror("DRM_IOCTL_MODE_CREATE_DUMB"); return 1;
    }

    fb_size = creq.size;

    struct drm_mode_map_dumb mreq = {0};
    mreq.handle = creq.handle;
    if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &mreq) < 0) {
        perror("DRM_IOCTL_MODE_MAP_DUMB"); return 1;
    }

    fb_map = mmap(0, fb_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mreq.offset);
    if (fb_map == MAP_FAILED) { perror("mmap"); return 1; }

    if (drmModeAddFB(fd, mode.hdisplay, mode.vdisplay, 24, 32, creq.pitch, creq.handle, &fb_id)) {
        perror("drmModeAddFB"); return 1;
    }

    if (drmModeSetCrtc(fd, enc->crtc_id, fb_id, 0, 0, &conn->connector_id, 1, &mode)) {
        perror("drmModeSetCrtc"); return 1;
    }

    atexit(cleanup);

    printf("\nRed circle: Keyboard 1 (WASD/arrows) - hold keys to move continuously\n");
    printf("Green circle: Keyboard 2 (WASD/arrows) - hold keys to move continuously\n");
    printf("Yellow circle: Keyboard 3 (WASD/arrows) - hold keys to move continuously\n");
    printf("P = grow circle, O = shrink circle\n");
    printf("R = increase charge (+), E = decrease charge (-)\n");
    printf("Like charges repel, opposite charges attract!\n");
    printf("Press Q or ESC on any keyboard to exit\n\n");

    while (!should_exit) {
        handle_input();      // Read keyboard events and update key states
        update_positions();  // Move circles based on held keys
        update_framebuffer(); // Render the scene
        usleep(16000);       // ~60fps
    }

    cleanup();
    return 0;
}