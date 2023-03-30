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
#define N_SCENE_FACES 6


iftSet *DDA3D(iftImage *img, int p1, int pn){
    int n;
    iftVoxel u1 = iftGetVoxelCoord(img, p1);
    iftVoxel un = iftGetVoxelCoord(img, pn);

    double dx, dy, dz;

    if (p1 == pn) n = 1;
    else{
        double Dx = un.x - u1.x;
        double Dy = un.y - u1.y;
        double Dz = un.z - u1.z;

        if ((fabs(Dx) >= fabs(Dy)) && (fabs(Dx) >= fabs(Dz))){
            n = (int)(fabs(Dx) + 1);
            dx = iftSign(Dx);
            dy = dx * ((double)Dy/Dx);
            dz = dx * ((double)Dz/Dx);
        }
        else if ((fabs(Dy) >= fabs(Dx)) && (fabs(Dy) >= fabs(Dz))){
            n = (int)(fabs(Dy) + 1);
            dy = iftSign(Dy);
            dx = dy * ((double)Dx/Dy);
            dz = dy * ((double)Dz/Dy);
        }
        else{
            n = (int)(fabs(Dz) + 1);
            dz = iftSign(Dz);
            dx = dz * ((double)Dx/Dz);
            dy = dz * ((double)Dy/Dz);
        }
    }

    iftSet *S = NULL;
    iftInsertSet(&S, p1);

    iftVoxel u,v;

    u.x = u1.x; u.y = u1.y; u.z = u1.z;

    for (int k = 1; k < n; k++){
        v.x = iftRound(u.x); v.y = iftRound(u.y); v.z = iftRound(u.z);
        
        int q = iftGetVoxelIndex(img, v);
        iftInsertSet(&S, q);

        u.x += dx;
        u.y += dy;
        u.z += dz;
    }

    return S;

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

iftVector createVector(int x, int y, int z){
    iftVector v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

iftPoint createPoint(int x, int y, int z){
    iftPoint p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
}

iftPoint subtractPoints(iftPoint p1, iftPoint p2){
    iftPoint r;
    r.x = p1.x - p2.x;
    r.y = p1.y - p2.y;
    r.z = p1.z - p2.z;

    return r;
}

typedef struct _face{
    iftVector normal_vector;
    iftPoint  center;
} Face;


Face *createSceneFaces(iftImage *img){
    Face *sceneFaces = (Face *)calloc(N_SCENE_FACES, sizeof(Face));

    sceneFaces[0].center        = createPoint(img->xsize - 1, img->ysize / 2, img->zsize/2);
    sceneFaces[0].normal_vector = createVector(1, 0, 0);

    sceneFaces[1].center        = createPoint(0, img->ysize / 2, img->zsize / 2);
    sceneFaces[1].normal_vector = createVector(-1, 0, 0);

    sceneFaces[2].center        = createPoint(img->xsize / 2, img->ysize - 1, img->zsize / 2);
    sceneFaces[2].normal_vector = createVector(0, 1, 0);

    sceneFaces[3].center        = createPoint(img->xsize / 2, 0, img->zsize / 2);
    sceneFaces[3].normal_vector = createVector(0, -1, 0);
    
    sceneFaces[4].center        = createPoint(img->xsize / 2, img->ysize / 2, img->zsize - 1);
    sceneFaces[4].normal_vector = createVector(0, 0, 1);

    sceneFaces[5].center        = createPoint(img->xsize / 2, img->ysize / 2, 0);
    sceneFaces[5].normal_vector = createVector(0, 0, -1);

    return sceneFaces;
}

iftVoxel getVoxelFromPoint(iftPoint p){
    iftVoxel u;
    u.x = (int)ceil(p.x);
    u.y = (int)ceil(p.y);
    u.z = (int)ceil(p.z);

    return u;
}


int main(int argc, char *argv[]){
    timer *tstart = NULL;
    int    MemDinInicial, MemDinFinal;

    MemDinInicial = iftMemoryUsed(1);

    if (argc != 6){
        printf("usage MIP: <P1> <P2> <P3> <P4> <P5>\n");
        printf("P1: input image (.scn)\n");
        printf("P2: tilt angle alpha\n");
        printf("P3: sping angle beta\n");
        printf("P4: .scn object mask (optional)\n");
        printf("P5: output .png image of the maximum intensity projection\n");
        exit(0);
    }
    tstart = iftTic();

    /* ----------------------- Coding Area -------------------------- */

    char filename[200];
    iftImage *img     = iftReadImageByExt(argv[1]);
    double alpha      = atof(argv[2]);
    double beta       = atof(argv[3]);
    
    int diagonal   = (int)sqrt(img->xsize * img->xsize + img->ysize * img->ysize + img->zsize * img->zsize);
    iftImage *mip  = iftCreateImage(diagonal, diagonal, 1);


    iftVector n                   = createVector(0, 0, 1);
    iftVector centerTranslation   = createVector(img->xsize / 2, img->ysize / 2, img->zsize / 2);
    iftVector diagonalTranslation = createVector(-diagonal / 2, -diagonal / 2, -diagonal / 2);


    iftMatrix *Rx    = iftRotationMatrix(IFT_AXIS_X, -alpha);
    iftMatrix *Ry    = iftRotationMatrix(IFT_AXIS_Y, -beta);
    iftMatrix *Tc    = iftTranslationMatrix(centerTranslation);
    iftMatrix *Td    = iftTranslationMatrix(diagonalTranslation);

    iftMatrix *Phi_r   = iftMultMatrices(Rx, Ry);
    iftMatrix *aux     = iftMultMatrices(Tc, Phi_r);
    iftMatrix *Phi_inv = iftMultMatrices(aux, Td);

    iftDestroyMatrix(&aux);
    
    iftPoint n_prime = iftTransformVector(Phi_r, n);
    Face *sceneFaces = createSceneFaces(img);

    int counter = 0;

    for (int i = 0; i < mip->n; i++){
        iftVoxel u  = iftGetVoxelCoord(mip, i);
        iftPoint p  = createPoint(u.x, u.y, -diagonal / 2);
        iftPoint p0 = iftTransformPoint(Phi_inv, p);


        double lambdaMin = IFT_INFINITY_DBL;
        double lambdaMax = IFT_INFINITY_DBL_NEG;

        /* Go through each face of the scene and solve for lambda */
        for (int f = 0; f < N_SCENE_FACES; f++){
            iftPoint faceCenter    = sceneFaces[f].center;
            iftVector normalVector = sceneFaces[f].normal_vector;
            iftPoint g             = subtractPoints(p0, faceCenter);

            double numerator   = (g.x * normalVector.x + g.y * normalVector.y + g.z * normalVector.z);
            double denominator = (n_prime.x * normalVector.x + n_prime.y * normalVector.y + n_prime.z * normalVector.z);
            double lambda      = -(numerator / denominator);

            
            iftPoint intersection = createPoint(iftRound(p0.x + lambda * n_prime.x), iftRound(p0.y + lambda * n_prime.y), iftRound(p0.z + lambda * n_prime.z));
            iftVoxel v            = getVoxelFromPoint(intersection);

            if (iftValidVoxel(img, v) && lambda > 0){
                if (lambda < lambdaMin) lambdaMin = lambda;
                if (lambda > lambdaMax) lambdaMax = lambda;
            }
        }

        if ((lambdaMin != IFT_INFINITY_DBL) && (lambdaMax != IFT_INFINITY_DBL_NEG)){
            iftPoint initialPoint = createPoint(iftRound(p0.x + lambdaMin * n_prime.x), iftRound(p0.y + lambdaMin * n_prime.y), iftRound(p0.z + lambdaMin * n_prime.z));
            iftPoint finalPoint   = createPoint(iftRound(p0.x + lambdaMax * n_prime.x), iftRound(p0.y + lambdaMax * n_prime.y), iftRound(p0.z + lambdaMax * n_prime.z));

            iftVoxel initialVoxel = getVoxelFromPoint(initialPoint);
            iftVoxel finalVoxel   = getVoxelFromPoint(finalPoint);

            int initialIndex = iftGetVoxelIndex(img, initialVoxel);
            int finalIndex   = iftGetVoxelIndex(img, finalVoxel);

            iftSet *S = DDA3D(img, initialIndex, finalIndex);
            
            int maximumIntensityValue = 0;
            int setSize               = iftSetSize(S);
            for (int k = 0; k < setSize; k++){
                int intensityIndex = iftRemoveSet(&S);
                int intensity      = img->val[intensityIndex];
                if (intensity >= maximumIntensityValue) maximumIntensityValue = intensity;
            }
            
            mip->val[i] = maximumIntensityValue;
            counter++;
            iftDestroySet(&S);
        }
    }

    radiometricEnhance(mip, mip, 0.5, 0.7);

    sprintf(filename, "%s", argv[5]);
    iftWriteImageByExt(mip,filename);
    
    

    iftDestroyImage(&img);
    iftDestroyImage(&mip);
    iftDestroyMatrix(&Rx);
    iftDestroyMatrix(&Ry);
    iftDestroyMatrix(&Phi_r);
    iftDestroyMatrix(&Tc);
    iftDestroyMatrix(&Td);
    iftDestroyMatrix(&Phi_inv);

    free(sceneFaces);
    /* -------------------- End of the coding area ----------------- */
        
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
        
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    
    return(0);
}