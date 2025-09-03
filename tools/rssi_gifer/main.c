#define __USE_MISC

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <jansson.h>
#include <cairo/cairo.h>
#include <math.h>
#include "gifenc.h"   // https://github.com/lecram/gifenc

#define NUM_BINS 50
#define WIDTH 640
#define HEIGHT 480
#define MAX_RSSI_VALUES 4096

// Parse JSON file and extract RSSI values
size_t load_rssi_values(const char *filename, int *rssi, size_t max_vals) {
    json_error_t error;
    json_t *root = json_load_file(filename, 0, &error);
    if (!root) {
        fprintf(stderr, "Error parsing %s: %s\n", filename, error.text);
        return 0;
    }

    json_t *measurements = json_object_get(root, "measurements");
    if (!json_is_array(measurements)) {
        fprintf(stderr, "Invalid JSON structure in %s\n", filename);
        json_decref(root);
        return 0;
    }

    size_t count = json_array_size(measurements);
    if (count > max_vals) count = max_vals;

    for (size_t i = 0; i < count; i++) {
        json_t *m = json_array_get(measurements, i);
        json_t *rssi_val = json_object_get(m, "rssi");
        if (json_is_integer(rssi_val)) {
            rssi[i] = (int)json_integer_value(rssi_val);
        } else {
            rssi[i] = 0;
        }
    }

    json_decref(root);
    return count;
}

// Find global min/max RSSI and max bin count across all files
void find_global_ranges(const char *indir, int *global_min, int *global_max, int *global_max_bin) {
    *global_min = 0;
    *global_max = -200;
    *global_max_bin = 0;
    
    char path[512];
    int rssi[MAX_RSSI_VALUES];
    
    for (int i = 1;; i++) {
        snprintf(path, sizeof(path), "%s/%d.json", indir, i);
        FILE *f = fopen(path, "r");
        if (!f) break;
        fclose(f);
        
        size_t count = load_rssi_values(path, rssi, MAX_RSSI_VALUES);
        if (count == 0) continue;
        
        // Find min/max for this file
        for (size_t j = 0; j < count; j++) {
            if (rssi[j] < *global_min) *global_min = rssi[j];
            if (rssi[j] > *global_max) *global_max = rssi[j];
        }
        
        // Calculate bins for this file to find max bin count
        int bins[NUM_BINS] = {0};
        int range = *global_max - *global_min;
        if (range == 0) range = 1; // Avoid division by zero
        
        for (size_t j = 0; j < count; j++) {
            int idx = (int)((rssi[j] - *global_min) * (double)NUM_BINS / (range + 1));
            if (idx < 0) idx = 0;
            if (idx >= NUM_BINS) idx = NUM_BINS - 1;
            bins[idx]++;
        }
        
        for (int j = 0; j < NUM_BINS; j++) {
            if (bins[j] > *global_max_bin) *global_max_bin = bins[j];
        }
    }
    
    // Ensure we have reasonable defaults
    if (*global_min == 0 && *global_max == -200) {
        *global_min = -100;
        *global_max = -30;
    }
    if (*global_max_bin == 0) *global_max_bin = 1;
}

// Draw histogram of RSSI values
void draw_histogram(cairo_t *cr, int *rssi, size_t count,
                    int min_rssi, int max_rssi, int global_max_bin, int frame_num)
{
    int bins[NUM_BINS] = {0};

    /* --- 1) build bins --- */
    int range = max_rssi - min_rssi;
    if (range == 0) range = 1; // Avoid division by zero
    
    for (size_t i = 0; i < count; ++i) {
        int v = rssi[i];
        if (v < min_rssi || v > max_rssi) continue;
        int idx = (int)((v - min_rssi) * (double)NUM_BINS / (range + 1));
        if (idx < 0) idx = 0;
        if (idx >= NUM_BINS) idx = NUM_BINS - 1;
        bins[idx]++;
    }

    /* --- 2) clear background --- */
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_paint(cr);

    /* --- layout --- */
    const int left = 80, right = 40, top = 60, bottom = 80;
    const double plot_w = (double)WIDTH - left - right;
    const double plot_h = (double)HEIGHT - top - bottom;

    /* --- 3) draw axes --- */
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_set_line_width(cr, 2.0);

    /* Y axis */
    cairo_move_to(cr, left, HEIGHT - bottom);
    cairo_line_to(cr, left, top);
    cairo_stroke(cr);

    /* X axis */
    cairo_move_to(cr, left, HEIGHT - bottom);
    cairo_line_to(cr, WIDTH - right, HEIGHT - bottom);
    cairo_stroke(cr);

    /* --- 4) compute bar positions --- */
    double bar_width = plot_w / (double)NUM_BINS;

    /* --- 5) draw bars --- */
    for (int i = 0; i < NUM_BINS; ++i) {
        double x = left + i * bar_width;
        
        double bar_h = 0;
        if (global_max_bin > 0) {
            bar_h = (double)bins[i] * plot_h / (double)global_max_bin;
        }

        double y = HEIGHT - bottom - bar_h;
        if (y < top) y = top; /* clamp */

        if (bar_h > 2) { // Only draw bars that are visible
            cairo_rectangle(cr, x + 1, y, bar_width - 2, bar_h);
            cairo_set_source_rgb(cr, 0.2, 0.4, 0.8); /* bar color */
            cairo_fill(cr);
            
            // Add border to bars
            cairo_rectangle(cr, x + 1, y, bar_width - 2, bar_h);
            cairo_set_source_rgb(cr, 0.1, 0.2, 0.4);
            cairo_set_line_width(cr, 0.5);
            cairo_stroke(cr);
        }
    }

    /* --- 6) labels and titles --- */
    cairo_select_font_face(cr, "Arial",
                           CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, 12);
    cairo_set_source_rgb(cr, 0, 0, 0);

    char buf[128];
    cairo_text_extents_t ext;

    /* Title */
    cairo_set_font_size(cr, 16);
    snprintf(buf, sizeof(buf), "RSSI Histogram - Frame %d", frame_num);
    cairo_text_extents(cr, buf, &ext);
    cairo_move_to(cr, WIDTH/2.0 - ext.width/2.0, 30);
    cairo_show_text(cr, buf);

    cairo_set_font_size(cr, 12);

    /* X axis labels */
    snprintf(buf, sizeof(buf), "%d", min_rssi);
    cairo_text_extents(cr, buf, &ext);
    cairo_move_to(cr, left - ext.width/2.0, HEIGHT - bottom + 20);
    cairo_show_text(cr, buf);

    int mid_rssi = (min_rssi + max_rssi) / 2;
    snprintf(buf, sizeof(buf), "%d", mid_rssi);
    cairo_text_extents(cr, buf, &ext);
    cairo_move_to(cr, left + plot_w/2.0 - ext.width/2.0, HEIGHT - bottom + 20);
    cairo_show_text(cr, buf);

    snprintf(buf, sizeof(buf), "%d", max_rssi);
    cairo_text_extents(cr, buf, &ext);
    cairo_move_to(cr, WIDTH - right - ext.width/2.0, HEIGHT - bottom + 20);
    cairo_show_text(cr, buf);

    /* Y axis labels */
    snprintf(buf, sizeof(buf), "0");
    cairo_text_extents(cr, buf, &ext);
    cairo_move_to(cr, left - ext.width - 10, HEIGHT - bottom + 4);
    cairo_show_text(cr, buf);

    snprintf(buf, sizeof(buf), "%d", global_max_bin);
    cairo_text_extents(cr, buf, &ext);
    cairo_move_to(cr, left - ext.width - 10, top + 4);
    cairo_show_text(cr, buf);

    /* Axis titles */
    cairo_set_font_size(cr, 14);
    
    // X axis title
    const char *xlabel = "RSSI (dBm)";
    cairo_text_extents(cr, xlabel, &ext);
    cairo_move_to(cr, left + plot_w/2.0 - ext.width/2.0, HEIGHT - 15);
    cairo_show_text(cr, xlabel);

    // Y axis title (rotated)
    const char *ylabel = "Count";
    cairo_save(cr);
    cairo_translate(cr, 20, HEIGHT/2.0);
    cairo_rotate(cr, -M_PI/2.0);
    cairo_text_extents(cr, ylabel, &ext);
    cairo_move_to(cr, -ext.width/2.0, ext.height/2.0);
    cairo_show_text(cr, ylabel);
    cairo_restore(cr);
}

// Convert Cairo surface to GIF frame with better color mapping
void surface_to_gif_frame(cairo_surface_t *surface, ge_GIF *gif) {
    unsigned char *data = cairo_image_surface_get_data(surface);
    int stride = cairo_image_surface_get_stride(surface);

    // Ensure surface data is flushed
    cairo_surface_flush(surface);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            unsigned char *p = data + y * stride + 4 * x;
            // Cairo uses BGRA format, map to palette
            unsigned char b = p[0];
            unsigned char g = p[1]; 
            unsigned char r = p[2];
            
            // Simple palette mapping
            if (r > 200 && g > 200 && b > 200) {
                gif->frame[y * WIDTH + x] = 1; // White
            } else if (r < 100 && g < 100 && b > 150) {
                gif->frame[y * WIDTH + x] = 2; // Blue bars
            } else {
                gif->frame[y * WIDTH + x] = 0; // Black/dark
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_dir> <output.gif>\n", argv[0]);
        return 1;
    }

    const char *indir = argv[1];
    const char *outfile = argv[2];

    // Find global ranges first
    int global_min, global_max, global_max_bin;
    find_global_ranges(indir, &global_min, &global_max, &global_max_bin);
    
    printf("Global RSSI range: %d to %d, max bin count: %d\n", 
           global_min, global_max, global_max_bin);

    // Create GIF with better palette
    ge_GIF *gif = ge_new_gif(
        outfile, WIDTH, HEIGHT,
        (uint8_t[]){
            0x00,0x00,0x00,  // Black
            0xFF,0xFF,0xFF,  // White  
            0x33,0x66,0xCC,  // Blue
            0x66,0x99,0xFF,  // Light blue
            0x99,0xCC,0xFF   // Very light blue
        }, 5, 0, 0
    );

    if (!gif) {
        fprintf(stderr, "Failed to create GIF file: %s\n", outfile);
        return 1;
    }

    char path[512];
    int rssi[MAX_RSSI_VALUES];
    int frame_count = 0;

    for (int i = 1;; i++) {
        snprintf(path, sizeof(path), "%s/%d.json", indir, i);
        FILE *f = fopen(path, "r");
        if (!f) break;
        fclose(f);

        printf("Processing frame %d: %s\n", frame_count + 1, path);

        size_t count = load_rssi_values(path, rssi, MAX_RSSI_VALUES);
        if (count == 0) {
            printf("Warning: No data in %s\n", path);
            continue;
        }

        cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, WIDTH, HEIGHT);
        if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
            fprintf(stderr, "Failed to create Cairo surface\n");
            break;
        }

        cairo_t *cr = cairo_create(surface);
        if (cairo_status(cr) != CAIRO_STATUS_SUCCESS) {
            fprintf(stderr, "Failed to create Cairo context\n");
            cairo_surface_destroy(surface);
            break;
        }

        draw_histogram(cr, rssi, count, global_min, global_max, global_max_bin, i);

        surface_to_gif_frame(surface, gif);
        ge_add_frame(gif, 5); // 500ms delay between frames

        cairo_destroy(cr);
        cairo_surface_destroy(surface);
        frame_count++;
    }

    printf("Created GIF with %d frames\n", frame_count);
    ge_close_gif(gif);
    
    if (frame_count == 0) {
        fprintf(stderr, "Warning: No frames were added to the GIF\n");
        return 1;
    }
    
    return 0;
}