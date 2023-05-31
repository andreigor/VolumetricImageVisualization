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
#define DEBUG 1

int getXSizeFromAdjRel(iftAdjRel *A){
    int min_x = IFT_INFINITY_INT;
    int max_x = IFT_INFINITY_INT_NEG;
    for (int i = 0; i < A->n; i++){
        if (A->dx[i] > max_x) max_x = A->dx[i];
        if (A->dx[i] < min_x) min_x = A->dx[i];
    }

    return (max_x - min_x + 1);
}

int getYSizeFromAdjRel(iftAdjRel *A){
    int min_y = IFT_INFINITY_INT;
    int max_y = IFT_INFINITY_INT_NEG;
    for (int i = 0; i < A->n; i++){
        if (A->dy[i] > max_y) max_y = A->dy[i];
        if (A->dy[i] < min_y) min_y = A->dy[i];
    }

    return (max_y - min_y + 1);
}

int getZSizeFromAdjRel(iftAdjRel *A){
    int min_z = IFT_INFINITY_INT;
    int max_z = IFT_INFINITY_INT_NEG;
    for (int i = 0; i < A->n; i++){
        if (A->dz[i] > max_z) max_z = A->dz[i];
        if (A->dz[i] < min_z) min_z = A->dz[i];
    }

    return (max_z - min_z + 1);
}

void printKernelBank(iftMMKernel *kernel_bank){
    // printf("n: %d\n", kernel_bank->A->n);
    // printf("Tamanho do kernel: %f\n", pow(kernel_bank->A->n, 1.0/3.0));

    char filename[200];

    int xsize = getXSizeFromAdjRel(kernel_bank->A);
    int ysize = getYSizeFromAdjRel(kernel_bank->A);
    int zsize = getZSizeFromAdjRel(kernel_bank->A);


    for (int k = 0; k < kernel_bank->nkernels; k++){
        iftImage *img = iftCreateImage(xsize, ysize, zsize);

        int dmin[3];
        dmin[0]=IFT_INFINITY_INT;
        dmin[1]=IFT_INFINITY_INT;
        dmin[2]=IFT_INFINITY_INT;

        // Finding minimum dx, dy and dz
        for (int j = 0; j < kernel_bank->A->n; j++){
            if (kernel_bank->A->dx[j]<dmin[0])
                dmin[0] = kernel_bank->A->dx[j];
            if (kernel_bank->A->dy[j]<dmin[1])
                dmin[1] = kernel_bank->A->dy[j];
            if (kernel_bank->A->dz[j]<dmin[2])
                dmin[2] = kernel_bank->A->dz[j];
        }

        float min_weight = IFT_INFINITY_FLT;
        float max_weight = IFT_INFINITY_FLT_NEG;

        for (int j=0; j < kernel_bank->A->n; j++){
            if(kernel_bank->weight[k][0].val[j] > max_weight)
                max_weight = kernel_bank->weight[k][0].val[j];
            
            if (kernel_bank->weight[k][0].val[j] < min_weight)
                min_weight = kernel_bank->weight[k][0].val[j];
        }

        for (int j=0; j < kernel_bank->A->n; j++) {
            kernel_bank->weight[k][0].val[j] = 255.0*(kernel_bank->weight[k][0].val[j] - min_weight)/(max_weight - min_weight);
        }

        iftVoxel u;
        for (int j=0; j < kernel_bank->A->n; j++) {
            u.x = kernel_bank->A->dx[j]-dmin[0];
            u.y = kernel_bank->A->dy[j]-dmin[1];
            u.z = kernel_bank->A->dz[j]-dmin[2];
            int p   = iftGetVoxelIndex(img,u);
            img->val[p] = iftRound(kernel_bank->weight[k][0].val[j]);
        }
        sprintf(filename,"kernel-%d.nii.gz", k);
        iftWriteImageByExt(img, filename);
        iftDestroyImage(&img);
    }
}



float meanArray(int n, float *arr){
    float mean = 0.0;
    for (int i = 0; i < n; i++){
        mean+= arr[i] / n;
    }
    return mean;
}

void normalizeKernelBank(iftMMKernel *kernel_bank){
    for (int k = 0; k < kernel_bank->nkernels; k++){
        float mean = meanArray(kernel_bank->A->n, kernel_bank->weight[k][0].val);

        // Subtract each value by mean
        for (int i = 0; i < kernel_bank->A->n; i++){
            kernel_bank->weight[k][0].val[i] -= mean;
        }
    }
}

iftMMKernel *createGrayScaleNormalizedRandomKernelBank(int n_kernels, float adj_radius){
    iftAdjRel *A             = iftSpheric(adj_radius);
    iftMMKernel *kernel_bank = iftCreateMMKernel(A, 1, n_kernels);

    for (int k = 0; k < n_kernels; k++){
        for (int i = 0; i < A->n; i++){
            kernel_bank->weight[k][0].val[i] = (float)rand()/RAND_MAX;
        }
    }

    // printf("Media do primeiro kernel antes: %f\n", meanArray(A->n, kernel_bank->weight[0][0].val));
    normalizeKernelBank(kernel_bank);
    // printf("Media do primeiro kernel depois: %f\n", meanArray(A->n, kernel_bank->weight[0][0].val));

    iftDestroyAdjRel(&A);

    return kernel_bank;
}

void ReLU(iftMImage *mimg){
    for (int b = 0; b < mimg->m; b++){
        for (int p = 0; p < mimg->n; p++){
            if (mimg->val[p][b] < 0) mimg->val[p][b] = 0;

        }
    }
}


int main(int argc, char *argv[]){
    timer *tstart = NULL;
    int    MemDinInicial, MemDinFinal;

    MemDinInicial = iftMemoryUsed(1);

    if ((argc != 5)){
        printf("usage create_kernels_and_extract_features: <P1> <P2> <P3> <P4>\n");
        printf("P1: input grayscale 3D image\n");
        printf("P2: number of kernels\n");
        printf("P3: adjacency radius\n");
        printf("P4: output activations (.mimg) \n");
        exit(0);
    }
    tstart = iftTic();

    /* ----------------------- Coding Area -------------------------- */

    iftImage *img    = iftReadImageByExt(argv[1]);
    int   n_kernels  = atoi(argv[2]);
    float adj_radius = atof(argv[3]); 

    iftMMKernel *kernel_bank = createGrayScaleNormalizedRandomKernelBank(n_kernels, adj_radius);
    iftMImage *mimg          = iftImageToMImage(img, GRAY_CSPACE);
    iftMImage *activ         = iftMMLinearFilter(mimg, kernel_bank);

    ReLU(activ);


    iftWriteMImage(activ, argv[4]);

    printf("N bands: %d\n", activ->m);
    printf("xsize: %d\n", activ->xsize);
    printf("ysize: %d\n", activ->ysize);
    printf("zsize: %d\n", activ->zsize);

    
    

    #ifdef DEBUG

    printf("Debugging: printing kernel bank...\n");
    printKernelBank(kernel_bank);

    #endif

    iftDestroyMMKernel(&kernel_bank);
    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
    iftDestroyMImage(&activ);

    /* -------------------- End of the coding area ----------------- */
        
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
        
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    
    return(0);
}