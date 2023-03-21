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

iftImage *getSagittalSlice(iftImage *img, int x, int perspective){
    // iftImage *sagittalSlice = iftCreateColorImage(img->ysize, img->zsize, 1, K2);
    iftImage *sagittalSlice = iftCreateImage(img->ysize, img->zsize, 1);

    iftVoxel u;

    int k = 0;
    // radiologist
    if (perspective == 0){         u.x = x;
        for (u.z = img->zsize - 1; u.z >= 0; u.z--){
            for (u.y = 0; u.y < img->ysize; u.y++){ 
                int p = iftGetVoxelIndex(img, u);
                sagittalSlice->val[k] = img->val[p];
                k++;
            }
        }
    }
    // neurologist
    else{
        u.x = (img->xsize - 1) - x;
        for (u.z = img->zsize - 1; u.z >= 0; u.z--){
            for (u.y = img->ysize - 1; u.y >= 0; u.y--){ 
                int p = iftGetVoxelIndex(img, u);
                sagittalSlice->val[k] = img->val[p];
                k++;
            }
        }
    }
    return sagittalSlice;
}


iftImage *getCoronalSlice(iftImage *img, int y, int perspective){
    // iftImage *coronalSlice = iftCreateColorImage(img->xsize, img->zsize, 1, K2);
    iftImage *coronalSlice = iftCreateImage(img->xsize, img->zsize, 1);

    iftVoxel u;

    int k = 0;
    // radiologist
    if (perspective == 0){ 
        u.y = y;
        for (u.z = img->zsize - 1; u.z >= 0; u.z--){
            for (u.x = 0; u.x < img->xsize; u.x++){
                int p = iftGetVoxelIndex(img, u);
                coronalSlice->val[k] = img->val[p];
                k++;
            }
        }
    }

    // neurologist
    else{ 
        u.y = (img->ysize - 1) - y;
        for (u.z = img->zsize - 1; u.z >= 0; u.z--){
                for (u.x = img->xsize - 1; u.x >= 0; u.x--){
                    int p = iftGetVoxelIndex(img, u);
                    coronalSlice->val[k] = img->val[p];
                    k++;
                }
            }
    }
    return coronalSlice;
}


iftImage *getAxialSlice(iftImage *img, int z, int perspective){
    // iftImage *axialSlice = iftCreateColorImage(img->xsize, img->ysize, 1, K2);
    iftImage *axialSlice = iftCreateImage(img->xsize, img->ysize, 1);

    iftVoxel u;

    int k = 0;
    // radiologist
    if (perspective == 0){
        u.z = z;
        for (u.y = 0; u.y < img->ysize; u.y++){
            for (u.x = 0; u.x < img->xsize; u.x++){
                int p = iftGetVoxelIndex(img, u);
                axialSlice->val[k] = img->val[p];
                k++;
            }
        }
    }

    // neurologist
    else{
        u.z = (img->zsize - 1) - z;
        for (u.y = 0; u.y < img->ysize; u.y++){
            for (u.x = img->xsize - 1; u.x >= 0; u.x--){
                int p = iftGetVoxelIndex(img, u);
                axialSlice->val[k] = img->val[p];
                k++;
            }
        }
    }
    return axialSlice;
}

void linearStretch(iftImage *img, int l1, int l2){
    for (int p = 0; p < img->n; p++){
        if      (img->val[p] < l1)      img->val[p] = (int)(K1);
        else if (img->val[p] >= l2)     img->val[p] = (int)(K2);
        else                            img->val[p] = (int)((K2 - K1) / (l2 - l1) * (img->val[p] - l1) + K1);
    }
}

void radiometricEnhance(iftImage *img, iftImage *slice, float window_percentage, float level_percentage){
    /* Adjustment of brightness and contrast, by linear stretching*/
    int lmax = iftMaximumValue(img);
    int lmin = iftMinimumValue(img);

    float window = window_percentage * (lmax - lmin);
    float level  = level_percentage * lmax;

    float l2 = level + window / 2;
    float l1 = level - window / 2;

    if (l1 < lmin) l1 = lmin;
    if (l2 > lmax) l2 = lmax;

    linearStretch(slice, l1, l2);
}

iftColor getHeatMapProportionalColor(int intensity, int maximumValue){
    float V = (float)(intensity)/maximumValue;
    V = (6 - 2) * V + 1;

    iftColor RGB;
    RGB.val[0] = maximumValue * iftMax(0, ( 3 - fabs(V - 4) - fabs(V - 5) ) / 2);
    RGB.val[1] = maximumValue * iftMax(0, ( 4 - fabs(V - 2) - fabs(V - 4) ) / 2);
    RGB.val[2] = maximumValue * iftMax(0, ( 3 - fabs(V - 1) - fabs(V - 2) ) / 2);

    iftColor YCbCr = iftRGBtoYCbCr(RGB, K2);
    return YCbCr;
}

iftImage *getColoredToSlice(iftImage *img){
    iftImage *coloredImage = iftCreateColorImage(img->xsize, img->ysize, img->zsize, K2);
    for (int p = 0; p < img->n; p++){
        iftColor YCbCr = getHeatMapProportionalColor(img->val[p], K2);
        coloredImage->val[p] = YCbCr.val[0];
        coloredImage->Cb[p]  = YCbCr.val[1];
        coloredImage->Cr[p]  = YCbCr.val[2];
    }

    return coloredImage;
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
    int perspective   = atoi(argv[5]);
    float window_p    = 1.0 - atof(argv[6]);
    float level_p     = 1.0 - atof(argv[7]);

    char filename[200];
    char *view_name = ((perspective == 0) ? "radiologist" :  "neurologist");

    if ((x < 0)||(x >= img->xsize))
        iftError("x-coordinate must be in [0,%d]", "main", img->xsize-1);
    if ((y < 0)||(y >= img->ysize))
        iftError("y-coordinate must be in [0,%d]", "main", img->ysize-1);
    if ((z < 0)||(z >= img->zsize))
        iftError("z-coordinate must be in [0,%d]", "main", img->zsize-1);
    if ((window_p < 0)||(window_p > 1.0))
        iftError("window percentage must be in [0,1]", "main");
    if ((level_p < 0)||(level_p > 1.0))
        iftError("level percentage must be in [0,1]", "main");
    if (perspective != 0 && perspective != 1)
        iftError("perspective must be 0 or 1", "main");
    

    iftImage *sagitalSlice = getSagittalSlice(img, x, perspective);
    iftImage *coronalSlice = getCoronalSlice(img, y, perspective);
    iftImage *axialSlice   = getAxialSlice(img, z, perspective);


    radiometricEnhance(img, sagitalSlice, window_p, level_p);
    radiometricEnhance(img, coronalSlice, window_p, level_p);
    radiometricEnhance(img, axialSlice,   window_p, level_p);


    iftImage *coloredSagital = getColoredToSlice(sagitalSlice);
    iftImage *coloredCoronal = getColoredToSlice(coronalSlice);
    iftImage *coloredAxial   = getColoredToSlice(axialSlice);


    sprintf(filename, "%s-sagital.png", view_name);
    iftWriteImageByExt(coloredSagital,filename);

    sprintf(filename, "%s-coronal.png", view_name);
    iftWriteImageByExt(coloredCoronal,filename);

    sprintf(filename, "%s-axial.png", view_name);
    iftWriteImageByExt(coloredAxial,filename);

    sprintf(filename, "%s-sagital_bw.png", view_name);
    iftWriteImageByExt(sagitalSlice,filename);

    sprintf(filename, "%s-coronal_bw.png", view_name);
    iftWriteImageByExt(coronalSlice,filename);

    sprintf(filename, "%s-axial_bw.png", view_name);
    iftWriteImageByExt(axialSlice,filename);

    iftDestroyImage(&sagitalSlice);
    iftDestroyImage(&coronalSlice);
    iftDestroyImage(&axialSlice);
    iftDestroyImage(&coloredSagital);
    iftDestroyImage(&coloredCoronal);
    iftDestroyImage(&coloredAxial);
    iftDestroyImage(&img);
    /* -------------------- End of the coding area ----------------- */
        
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
        
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    
    return(0);




    return 0;
}