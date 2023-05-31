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

void filterBasedNormalization(iftImage *img, float min, float max){
    for (int i = 0; i < img->n; i++){
        img->val[i] = (int)(K * (((float)(img->val[i] - min)) / (max - min)));
    }
}


int main(int argc, char *argv[]){
    timer *tstart = NULL;
    int    MemDinInicial, MemDinFinal;

    MemDinInicial = iftMemoryUsed(1);

    if ((argc != 3)){
        printf("usage model_selection.c: <P1> <P2>\n");
        printf("P1: folder with train activations .mimg\n");
        printf("P2: output folder with normalized activations (.scn)\n");
        exit(0);
    }
    tstart = iftTic();

    char filename[200];

    /* ----------------------- Coding Area -------------------------- */

    iftFileSet *train_set = iftLoadFileSetFromDirBySuffix(argv[1], ".mimg", 1);
    iftMakeDir(argv[2]);

    sprintf(filename, "%s/%s.mimg", argv[1], iftFilename(train_set->files[0]->path, ".mimg"));
    iftMImage *activ = iftReadMImage(filename);
    iftImage  *img;

    int n_filters = activ->m;

    float *max_per_band = (float *)calloc(n_filters, sizeof(float));
    float *min_per_band = (float *)calloc(n_filters, sizeof(float));

    for (int i = 0; i < n_filters; i++){
        max_per_band[i] = IFT_INFINITY_FLT_NEG;
        min_per_band[i] = IFT_INFINITY_FLT;
    }

    for (int p = 0; p < activ->n; p++){
        for (int b = 0; b < activ->m; b++){
            if (activ->val[p][b] > max_per_band[b]) max_per_band[b] = activ->val[p][b];
            if (activ->val[p][b] < min_per_band[b]) min_per_band[b] = activ->val[p][b];
        }
    }

    iftDestroyMImage(&activ);

    for (int i = 1; i < train_set->n; i++){
        sprintf(filename, "%s/%s.mimg", argv[1], iftFilename(train_set->files[i]->path, ".mimg"));
        activ = iftReadMImage(filename);

        for (int p = 0; p < activ->n; p++){
            for (int b = 0; b < activ->m; b++){
                if (activ->val[p][b] > max_per_band[b]) max_per_band[b] = activ->val[p][b];
                if (activ->val[p][b] < min_per_band[b]) min_per_band[b] = activ->val[p][b];
            }
        }
        iftDestroyMImage(&activ);
    }

    for (int i = 0; i < train_set->n; i++){
        sprintf(filename, "%s/%s.mimg", argv[1], iftFilename(train_set->files[i]->path, ".mimg"));
        activ = iftReadMImage(filename);

        for (int b = 0; b < activ->m; b++){
            img = iftCreateImage(activ->xsize, activ->ysize, activ->zsize);
            for (int z = 0; z < activ->zsize; z++){
                for (int y = 0; y < activ->ysize; y++){
                    for (int x = 0; x < activ->xsize; x++){
                        iftVoxel u;
                        u.x = x;
                        u.y = y;
                        u.z = z;
                        int p = iftGetVoxelIndex(activ, u);
                        img->val[p] = activ->val[p][b];
                    }
                }
            }
            filterBasedNormalization(img, min_per_band[b], max_per_band[b]);

            printf("Banda: %d, max: %f, min: %f\n", b, max_per_band[b], min_per_band[b]);
            sprintf(filename, "%s/img_%d_filter_%d.scn", argv[2], i, b);
            iftWriteImageByExt(img, filename);
            iftDestroyImage(&img);
        }
        
        iftDestroyMImage(&activ);
    }

    free(max_per_band);
    free(min_per_band);
    iftDestroyFileSet(&train_set);


    /* -------------------- End of the coding area ----------------- */
        
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
        
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    
    return(0);
}