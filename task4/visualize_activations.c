/*
Computing Institute - Unicamp
Volumetric Image Visualization - MO815 - 1s2023
Professor: Alexandre Falcão
Student:   André Nóbrega
Creates a bank of uniformely random kernels.
*/


#include <stdio.h>
#include <stdlib.h>

#include "ift.h"

#define K 4095


void minMaxNormalization(iftImage *img){
    int min = IFT_INFINITY_INT;
    int max = IFT_INFINITY_INT_NEG;

    for (int i = 0; i < img->n; i++){
        if (img->val[i] < min) min = img->val[i];
        if (img->val[i] > max) max = img->val[i];
    }

    for (int i = 0; i < img->n; i++){
        img->val[i] = (int)(K * (((float)(img->val[i] - min)) / (max - min)));
    }
}


int main(int argc, char *argv[]){
    timer *tstart = NULL;
    int    MemDinInicial, MemDinFinal;

    MemDinInicial = iftMemoryUsed(1);

    if ((argc != 3)){
        printf("usage visualize_activations: <P1> <P2>\n");
        printf("P1: input activations .mimg\n");
        printf("P2: output folder with activations (.scn)\n");
        exit(0);
    }
    tstart = iftTic();

    char filename[200];

    /* ----------------------- Coding Area -------------------------- */

    iftMImage *mimg = iftReadMImage(argv[1]);


    for (int b = 0; b < mimg->m; b++){
        iftImage *activ = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);

        for (int z = 0; z < mimg->zsize; z++){
            for (int y = 0; y < mimg->ysize; y++){
                for (int x = 0; x < mimg->xsize; x++){
                    iftVoxel u;
                    u.x = x;
                    u.y = y;
                    u.z = z;
                    int p = iftGetVoxelIndex(mimg, u);
                    activ->val[p] = mimg->val[p][b];
                }
            }
        }
        minMaxNormalization(activ);

        sprintf(filename, "%s/activ_%d.scn", argv[2], b);
        iftWriteImageByExt(activ, filename);

        iftDestroyImage(&activ);
    }


    iftDestroyMImage(&mimg);

    /* -------------------- End of the coding area ----------------- */
        
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
        
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    
    return(0);
}