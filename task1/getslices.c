/*
Computing Institute - Unicamp
Volumetric Image Visualization - MO815 - 1s2023
Professor: Alexandre Falcão
Student:   André Nóbrega
*/

#include <stdio.h>
#include <string.h>
#include "ift.h"

#define K1 0
#define K2 65535

iftImage *getSagittalSlice(iftImage *img, int x){
    iftImage *sagittalSlice = iftCreateColorImage(img->zsize, img->ysize, 1, 3);
    iftVoxel u;
    u.x = x;
    int k = 0;
    
    for (u.y = 0; u.y < img->ysize; u.y++){
        for (u.z = 0; u.z < img->zsize; u.z++){
            int p = iftGetVoxelIndex(img, u);
            sagittalSlice->val[k] = img->val[p];
            k++;
        }
    }
    return sagittalSlice;
}


iftImage *getCoronalSlice(iftImage *img, int y){
    iftImage *coronalSlice = iftCreateColorImage(img->xsize, img->zsize, 1, 3);
    iftVoxel u;
    u.y = y;
    int k = 0;

    for (u.z = 0; u.z < img->zsize; u.z++){
        for (u.x = 0; u.x < img->xsize; u.x++){
            int p = iftGetVoxelIndex(img, u);
            coronalSlice->val[k] = img->val[p];
            k++;
        }
    }
    return coronalSlice;
}


iftImage *getAxialSlice(iftImage *img, int z){
    iftImage *axialSlice = iftCreateColorImage(img->xsize, img->ysize, 1, 3);
    iftVoxel u;
    u.z = z;
    int k = 0;

    for (u.y = 0; u.y < img->ysize; u.y++){
        for (u.x = 0; u.x < img->xsize; u.x++){
            int p = iftGetVoxelIndex(img, u);
            axialSlice->val[k] = img->val[p];
            k++;
        }
    }
    return axialSlice;
}

void linearStretch(iftImage *img, int l1, int l2){
    for (int p = 0; p < img->n; p++){
        if (img->val[p] < l1) img->val[p] = K1;
        else if (img->val[p] >= l2) img->val[p] = K2;
        else img->val[p] = (K2 - K1) / (l2 - l1) * (img->val[p] - l1) + K1;
    }
}

void radiometricEnhance(iftImage *img, iftImage *slice, float window_percentage, float level_percentage){
    /* Adjustment of brightness and contrast, by linear stretching*/
    int lmax = iftMaximumValue(img);
    int lmin = iftMinimumValue(img);

    int window = window_percentage * (lmax - lmin);
    int level  = level_percentage * lmax;

    int l2 = level + window / 2;
    int l1 = level - window / 2;

    linearStretch(slice, l1, l2);
}

void applyColoringToSlice(iftImage *img){
    iftColorTable *ctb = iftBlueToRedColorTable(K2); // K2 = 65535 (2^16 - 1)
    iftConvertRGBColorTableToYCbCrColorTable(ctb, 255);

    for (int p = 0; p < img->n; p++){
        img->val[p] = ctb->color[img->val[p]].val[0];
        img->Cb[p]  = ctb->color[img->val[p]].val[1];
        img->Cr[p]  = ctb->color[img->val[p]].val[2];
    }
}


int main(int argc, char *argv[]){
    timer *tstart = NULL;
    int    MemDinInicial, MemDinFinal;

    MemDinInicial = iftMemoryUsed(1);

    if (argc != 8){
        printf("usage getslices: <P1> <P2> <P3> <P4> <P5> <P6> <P7>\n");
        printf("P1: input image (assumed to be acquired with axial slices along the z axis) \n");
        printf("P2: x-coordinate for sagital slice\n");
        printf("P3: y-coordinate for coronal slice\n");
        printf("P4: z-coordinate for axial slice\n");
        printf("P5: option 0/1 for point-of-view of radiologists/neurologists, respectively\n");
        printf("P6: window size for linear stretching, as percentage of intensity range\n");
        printf("P7: level for linear stretching, as percentage of intensity range\n");
        exit(0);
    }
    tstart = iftTic();

    /* ----------------------- Coding Area -------------------------- */

    iftImage *img     = iftReadImageByExt(argv[1]);
    int x             = atoi(argv[2]);
    int y             = atoi(argv[3]);
    int z             = atoi(argv[4]);
    int point_of_view = atoi(argv[5]);
    float window_p    = atof(argv[6]);
    float level_p     = atof(argv[7]);
    char filename[200];
    char *view_name = ((point_of_view == 0) ? "radiologist" :  "neurologist");

    if ((x < 0)||(x >= img->xsize))
        iftError("x-coordinate must be in [0,%d]", "main", img->xsize-1);
    if ((y < 0)||(y >= img->ysize))
        iftError("y-coordinate must be in [0,%d]", "main", img->ysize-1);
    if ((z < 0)||(z >= img->zsize))
        iftError("z-coordinate must be in [0,%d]", "main", img->zsize-1);
    

    iftImage *sagitalSlice = getSagittalSlice(img, x);
    iftImage *coronalSlice = getCoronalSlice(img, y);
    iftImage *axialSlice   = getAxialSlice(img, z);


    radiometricEnhance(img, sagitalSlice, window_p, level_p);
    radiometricEnhance(img, coronalSlice, window_p, level_p);
    radiometricEnhance(img, axialSlice, window_p, level_p);


    applyColoringToSlice(axialSlice);

    sprintf(filename, "%s-sagital.png", view_name);
    iftWriteImageByExt(sagitalSlice,filename);

    sprintf(filename, "%s-coronal.png", view_name);
    iftWriteImageByExt(coronalSlice,filename);

    sprintf(filename, "%s-axial.png", view_name);
    iftWriteImageByExt(axialSlice,filename);

    iftDestroyImage(&sagitalSlice);
    iftDestroyImage(&coronalSlice);
    iftDestroyImage(&axialSlice);
    iftDestroyImage(&img);
    /* -------------------- End of the coding area ----------------- */
        
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
        
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    
    return(0);




    return 0;
}