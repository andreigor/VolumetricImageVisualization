/*
Computing Institute - Unicamp
Volumetric Image Visualization - MO815 - 1s2023
Professor: Alexandre Falcão
Student:   André Nóbrega
*/


#include <stdio.h>
#include "ift.h"

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


iftVector subtractPointsAsVector(iftPoint p1, iftPoint p2){
    iftVector r;
    r.x = p1.x - p2.x;
    r.y = p1.y - p2.y;
    r.z = p1.z - p2.z;

    return r;
}

int trilinearInterpolation(iftImage *scene, iftPoint p){
    iftVoxel v;
    v.x = iftRound(p.x); v.y = iftRound(p.y); v.z = iftRound(p.z);
    
    if (!iftValidVoxel(scene, v)) return 0;

    int x_floor, x_ceil, y_floor, y_ceil, z_floor, z_ceil;
    x_floor = (int)floor(p.x+ 0.0001) ; x_ceil = (int)ceil(p.x+ 0.0001) ;
    y_floor = (int)floor(p.y + 0.0001); y_ceil = (int)ceil(p.y+ 0.0001) ;
    z_floor = (int)floor(p.z + 0.0001); z_ceil = (int)ceil(p.z+ 0.0001) ;

    iftVoxel q1, q2, q3, q4, q5, q6, q7, q8;
    q1.x = x_floor; q1.y = y_ceil;  q1.z = z_floor;
    q2.x = x_ceil;  q2.y = y_ceil;  q2.z = z_floor;
    q3.x = x_floor; q3.y = y_floor; q3.z = z_floor;
    q4.x = x_ceil;  q4.y = y_floor; q4.z = z_floor;
    q5.x = x_floor; q5.y = y_ceil;  q5.z = z_ceil;
    q6.x = x_ceil;  q6.y = y_ceil;  q6.z = z_ceil;
    q7.x = x_floor; q7.y = y_floor; q7.z = z_ceil;
    q8.x = x_ceil;  q8.y = y_floor; q8.z = z_ceil;

    iftPoint q24, q68, q57, q13;
    q24.x = x_ceil; q24.y = p.y; q24.z = z_floor;
    q68.x = x_ceil; q68.y = p.y; q68.z = z_ceil;
    q13.x = x_floor; q13.y = p.y; q13.z = z_floor;
    q57.x = x_floor; q57.y = p.y; q57.z = z_ceil;

    iftPoint q1357, q2468;
    
    q1357.x = x_floor; q1357.y = p.y; q1357.z = p.z;
    q2468.x = x_ceil; q2468.y = p.y; q2468.z = p.z;

    float Iq2468, Iq1357, Iq24, Iq68, Iq13, Iq57;

    Iq24 = (float)scene->val[iftGetVoxelIndex(scene, q2)]*(p.y - q4.y) + (float)scene->val[iftGetVoxelIndex(scene, q4)]*(q2.y - p.y);
    Iq68 = (float)scene->val[iftGetVoxelIndex(scene, q6)]*(p.y - q8.y) + (float)scene->val[iftGetVoxelIndex(scene, q8)]*(q6.y - p.y);
    Iq13 = (float)scene->val[iftGetVoxelIndex(scene, q1)]*(p.y - q3.y) + (float)scene->val[iftGetVoxelIndex(scene, q3)]*(q1.y - p.y);
    Iq57 = (float)scene->val[iftGetVoxelIndex(scene, q5)]*(p.y - q7.y) + (float)scene->val[iftGetVoxelIndex(scene, q7)]*(q5.y - p.y);

    Iq2468 = Iq68 * (p.z - q24.z) + Iq24 * (q68.z - p.z);
    Iq1357 = Iq57 * (p.z - q13.z) + Iq13 * (q57.z - p.z);

    int I = (int)(Iq2468 * (p.x - q1357.x) + Iq1357 * (q2468.x - p.x));

    return I;
}


iftImage *GetSlice(iftImage *scene, iftPoint p0, iftVector n_prime){
    
    /*Create image and define normal vector*/
    int diagonal    = (int)sqrt(scene->xsize * scene->xsize + scene->ysize * scene->ysize + scene->zsize * scene->zsize);
    iftImage *slice = iftCreateImage(diagonal, diagonal, 1);

    /*Define translation matrix*/
    iftVector pointTranslation    = createVector(p0.x, p0.y, p0.z);
    iftVector diagonalTranslation = createVector(-diagonal / 2, -diagonal / 2, diagonal / 2);
    iftMatrix *Tp                 = iftTranslationMatrix(pointTranslation);
    iftMatrix *Td                 = iftTranslationMatrix(diagonalTranslation);

    /*Define rotation matrix*/
    double alpha     = atan2(n_prime.y, n_prime.z);
    double beta      = asin((n_prime.x));

    // printf("Alpha: %lf, Beta: %lf\n", alpha, beta);
    iftMatrix *Rx    = iftRotationMatrix(IFT_AXIS_X, -alpha * 180 / IFT_PI);
    iftMatrix *Ry    = iftRotationMatrix(IFT_AXIS_Y, beta * 180 / IFT_PI);

    /*Get final transformation matrices*/
    iftMatrix *Phi_r   = iftMultMatrices(Rx, Ry);

    iftMatrix *aux     = iftMultMatrices(Tp, Phi_r);
    iftMatrix *Phi     = iftMultMatrices(aux, Td);
    iftDestroyMatrix(&aux);

    /*For each pixel of slice*/
    for (int i = 0; i < slice->n; i++){
        /*Get transformed pixel*/
        iftVoxel u       = iftGetVoxelCoord(slice, i);
        iftPoint p       = createPoint(u.x, u.y, -diagonal/2);
        iftPoint p_prime = iftTransformPoint(Phi, p);

        slice->val[i] = trilinearInterpolation(scene, p_prime);
        // iftVoxel v;
        // v.x = iftRound(p_prime.x); v.y = iftRound(p_prime.y); v.z = iftRound(p_prime.z);

        // /*If valid voxel, interpolate and atribute to slice*/
        // if(iftValidVoxel(scene, v)){
        //     int q         = iftGetVoxelIndex(scene, v);
        //     slice->val[i] = scene->val[q];
        // }
    }

    iftDestroyMatrix(&Rx);
    iftDestroyMatrix(&Ry);
    iftDestroyMatrix(&Tp);
    iftDestroyMatrix(&Td);

    iftDestroyMatrix(&Phi_r);
    iftDestroyMatrix(&Phi);


    return slice;
}

void putSliceInImage(iftImage *img, iftImage *slice, int z){
    iftVoxel u, v;
    u.z = z;
    v.z = 0;
    for (int x = 0; x < img->xsize; x++){
        for (int y = 0; y < img->ysize; y++){
            u.x = x; v.x = x;
            u.y = y; v.y = y;
            int p = iftGetVoxelIndex(img, u);
            int q = iftGetVoxelIndex(slice, v);
            img->val[p] = slice->val[q];
        }
    }
}

iftPoint addPoint(iftPoint p0, iftPoint p1){
    iftPoint p;
    p.x = p0.x + p1.x;
    p.y = p0.y + p1.y;
    p.z = p0.z + p1.z;

    return p;
}


int main(int argc, char *argv[]){
    timer *tstart = NULL;
    int    MemDinInicial, MemDinFinal;

    MemDinInicial = iftMemoryUsed(1);

    if ((argc != 10)){
        printf("usage reslicing: <P1> <P2> <P3> <P4> <P5> <P6> <P7> <P8> <P9>\n");
        printf("P1: input image (.scn)\n");
        printf("P2: P0.x\n");
        printf("P3: P0.y\n");
        printf("P4: P0.z\n");
        printf("P5: P1.x\n");
        printf("P6: P1.y\n");
        printf("P7: P1.z\n");
        printf("P8: number of axial slices of the new scene\n");
        printf("P9: output .scn scene\n");
        exit(0);
    }
    tstart = iftTic();

    /* ----------------------- Coding Area -------------------------- */

    char filename[200];
    iftImage *scene   = iftReadImageByExt(argv[1]);
    int p0x           = atoi(argv[2]);
    int p0y           = atoi(argv[3]);
    int p0z           = atoi(argv[4]);
    int p1x           = atoi(argv[5]);
    int p1y           = atoi(argv[6]);
    int p1z           = atoi(argv[7]);
    int n_slices      = atoi(argv[8]);

    printf("xsize: %d, ysize: %d, zsize: %d\n", scene->xsize, scene->ysize, scene->zsize);
    
    
    iftPoint p0       = createPoint(p0x, p0y, p0z);
    iftPoint p1       = createPoint(p1x, p1y, p1z);
    iftVector n_prime = subtractPointsAsVector(p1, p0);
    float     lambda  = sqrt((n_prime.x * n_prime.x) + (n_prime.y * n_prime.y) + (n_prime.z * n_prime.z))/((float)n_slices);
    n_prime           = iftNormalizeVector(n_prime);

    int diagonal   = (int)sqrt(scene->xsize * scene->xsize + scene->ysize * scene->ysize + scene->zsize * scene->zsize);
    iftImage *img  = iftCreateImage(diagonal, diagonal, n_slices);
    iftImage *slice = GetSlice(scene, p0, n_prime);

    putSliceInImage(img, slice, 0);
    iftDestroyImage(&slice);

    for (int k = 1; k < n_slices; k++){
        p0.x = p0.x + lambda * n_prime.x;
        p0.y = p0.y + lambda * n_prime.y;
        p0.z = p0.z + lambda * n_prime.z;

        slice       = GetSlice(scene, p0, n_prime);
        putSliceInImage(img, slice, k);
        iftDestroyImage(&slice);
    }

    sprintf(filename, "%s", argv[9]);
    iftWriteImageByExt(img, filename);

    iftDestroyImage(&scene);
    iftDestroyImage(&img);




    /* -------------------- End of the coding area ----------------- */
        
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
        
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    
    return(0);
}