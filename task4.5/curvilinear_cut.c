#include <stdio.h>
#include <stdlib.h>

#include "ift.h"

#define CLOSE_RADIUS 100
#define N_SCENE_FACES 6
#define N_SCENE_VERTEX 8
#define N_SCENE_EDGES 12
#define K1 0
#define K2 65535

#define BRIGHTNESS 0.3
#define CONTRAST 0.3
// #define DEBUG 1

iftImage *getEnvelopFromSegmentationMask(iftImage *mask, int radius){
    iftImage *envelop = iftCloseBin(mask, radius);
    return envelop;
}



iftImage *binMultipleObjectMask(iftImage *mask){
    iftImage *img = iftCreateImageFromImage(mask);

    for (int p = 0; p < img->n; p++){
        if (mask->val[p] != 0) img->val[p] = 1;
        else                   img->val[p] = 0;
    }
    return img;
}

int squaredEuclideanDistance(iftVoxel u, iftVoxel v){
    int distance = (((u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y) + (u.z - v.z) * (u.z - v.z)));
    return distance;
}

iftImage *getEDTFromEnvelop(iftImage *envelop){
    iftImage *cost = iftCreateImage(envelop->xsize, envelop->ysize, envelop->zsize);
    iftImage *root = iftCreateImage(envelop->xsize, envelop->ysize, envelop->zsize);
    iftAdjRel *A   = iftSpheric(sqrt(3.0));
    iftAdjRel *B   = iftSpheric(1.0);

    iftSet *S      = iftObjectBorderSet(envelop, B);

    float diagonal = (envelop->xsize) * (envelop->xsize) + (envelop->ysize) * (envelop->ysize) + (envelop->zsize) * (envelop->zsize);
    iftGQueue *Q   = iftCreateGQueue(diagonal, envelop->n, cost->val);


    for (int p = 0; p < envelop->n; p++){
        if (envelop->val[p] != 0) cost->val[p] = IFT_INFINITY_INT;
    }

    while (S != NULL){
        int p = iftRemoveSet(&S);
        cost->val[p] = 0;
        root->val[p] = p;
        iftInsertGQueue(&Q, p);
    }

    while(!iftEmptyGQueue(Q)){
        int p               = iftRemoveGQueue(Q);
        iftVoxel u          = iftGetVoxelCoord(envelop, p);
        iftVoxel root_voxel = iftGetVoxelCoord(envelop, root->val[p]);

        for (int i = 0; i < A->n; i++){
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftValidVoxel(envelop, v)){
                int q = iftGetVoxelIndex(envelop, v);

                if ((cost->val[q] > cost->val[p]) && (envelop->val[q] != 0)){
                    int tmp = squaredEuclideanDistance(v, root_voxel);

                    if (tmp < cost->val[q]){
                        if (Q->L.elem[q].color == IFT_GRAY)
                            iftRemoveGQueueElem(Q, q);
                    
                        cost->val[q] = tmp;
                        root->val[q] = root->val[p];
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }


    iftDestroyImage(&root);
    iftDestroyGQueue(&Q);
    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);

    return cost;


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


typedef struct _face{
    iftVector normal_vector;
    iftPoint  center;
} Face;

typedef struct _vertex{
    iftPoint vertex;
} Vertex;

typedef struct _edge{
    Vertex v1, v2;
} Edge;

Edge *createSceneEdges(Vertex *sceneVertex){
    Edge *sceneEdges = (Edge *)calloc(N_SCENE_EDGES, sizeof(Edge));

    sceneEdges[0].v1 = sceneVertex[0];
    sceneEdges[0].v2 = sceneVertex[1];

    sceneEdges[1].v1 = sceneVertex[1];
    sceneEdges[1].v2 = sceneVertex[3];
    
    sceneEdges[2].v1 = sceneVertex[0];
    sceneEdges[2].v2 = sceneVertex[2];

    sceneEdges[3].v1 = sceneVertex[2];
    sceneEdges[3].v2 = sceneVertex[3];

    sceneEdges[4].v1 = sceneVertex[0];
    sceneEdges[4].v2 = sceneVertex[4];

    sceneEdges[5].v1 = sceneVertex[1];
    sceneEdges[5].v2 = sceneVertex[5];

    sceneEdges[6].v1 = sceneVertex[2];
    sceneEdges[6].v2 = sceneVertex[6];

    sceneEdges[7].v1 = sceneVertex[3];
    sceneEdges[7].v2 = sceneVertex[7];

    sceneEdges[8].v1 = sceneVertex[4];
    sceneEdges[8].v2 = sceneVertex[5];

    sceneEdges[9].v1 = sceneVertex[4];
    sceneEdges[9].v2 = sceneVertex[6];

    sceneEdges[10].v1 = sceneVertex[5];
    sceneEdges[10].v2 = sceneVertex[7];

    sceneEdges[11].v1 = sceneVertex[6];
    sceneEdges[11].v2 = sceneVertex[7];

    return sceneEdges;
}

Vertex *createSceneVertex(iftImage *img){
    Vertex *sceneVertex = (Vertex *)calloc(N_SCENE_VERTEX, sizeof(Vertex));

    sceneVertex[0].vertex = createPoint(0, 0, 0);
    sceneVertex[1].vertex = createPoint(img->xsize - 1, 0, 0);
    sceneVertex[2].vertex = createPoint(0, img->ysize - 1, 0);
    sceneVertex[3].vertex = createPoint(img->xsize - 1, img->ysize - 1, 0);
    sceneVertex[4].vertex = createPoint(0, 0, img->zsize - 1);
    sceneVertex[5].vertex = createPoint(img->xsize - 1, 0, img->zsize - 1);
    sceneVertex[6].vertex = createPoint(0, img->ysize - 1, img->zsize - 1);
    sceneVertex[7].vertex = createPoint(img->xsize - 1, img->ysize - 1, img->zsize - 1);

    return sceneVertex;
}

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

iftPoint subtractPoints(iftPoint p1, iftPoint p2){
    iftPoint r;
    r.x = p1.x - p2.x;
    r.y = p1.y - p2.y;
    r.z = p1.z - p2.z;

    return r;
}

iftVoxel getVoxelFromPoint(iftPoint p){
    iftVoxel u;
    u.x = (int)ceil(p.x);
    u.y = (int)ceil(p.y);
    u.z = (int)ceil(p.z);

    return u;
}


iftPoint findIsoSurfacePoint(iftImage *edt, float depth, iftPoint p1, iftPoint pn){
    int n;

    iftVoxel initialVoxel = getVoxelFromPoint(p1);
    iftVoxel finalVoxel   = getVoxelFromPoint(pn);

    double dx, dy, dz;

    if ((initialVoxel.x == finalVoxel.x) && (initialVoxel.y == finalVoxel.y) && (initialVoxel.z == finalVoxel.z)) n = 1;
    else{
        double Dx = pn.x - p1.x;
        double Dy = pn.y - p1.y;
        double Dz = pn.z - p1.z;

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
    

    iftPoint p_prime  = createPoint(initialVoxel.x, initialVoxel.y, initialVoxel.z);
    int p_prime_index = iftGetVoxelIndex(edt, initialVoxel);
    iftVoxel u        = iftGetVoxelCoord(edt, p_prime_index);
    iftAdjRel *A      = iftSpheric(1.0);

    for (int i = 0; i < A->n; i++){
        iftVoxel v = iftGetAdjacentVoxel(A, u, i);
        if (iftValidVoxel(edt, v)){
            int q = iftGetVoxelIndex(edt, v);
            if (((depth - 0.8) < sqrt(edt->val[q])) && ((depth + 0.8) > sqrt(edt->val[q])))
                return p_prime;
        }
    }
    
    for (int k = 1; k < n; k++){
        p_prime.x += dx;
        p_prime.y += dy;
        p_prime.z += dz;
        
        iftVoxel p_prime_voxel = getVoxelFromPoint(p_prime);
        p_prime_index          = iftGetVoxelIndex(edt, p_prime_voxel);

        for (int i = 0; i < A->n; i++){
            iftVoxel v = iftGetAdjacentVoxel(A, p_prime_voxel, i);
            if (iftValidVoxel(edt, v)){
                int q = iftGetVoxelIndex(edt, v);
                if (((depth - 0.8) < sqrt(edt->val[q])) && ((depth + 0.8) > sqrt(edt->val[q])))
                    return p_prime;

            }
        }
    }

    p_prime.x = -10;

    return p_prime;
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

iftVector subtractPointsAsVector(iftPoint p1, iftPoint p2){
    iftVector r;
    r.x = p1.x - p2.x;
    r.y = p1.y - p2.y;
    r.z = p1.z - p2.z;

    return r;
}

void drawPoint(iftImage *img, iftVoxel u, iftAdjRel *A){
    for (int i = 0; i < A->n; i++){
        iftVoxel v = iftGetAdjacentVoxel(A, u, i);
        if (iftValidVoxel(img, v)){
            int q = iftGetVoxelIndex(img, v);
            img->val[q] = K2;
        }
    }
}

void draw2DLine(iftImage *img, iftPoint p1, iftPoint pn){
    int n;

    iftVoxel initialVoxel = getVoxelFromPoint(p1);
    iftVoxel finalVoxel   = getVoxelFromPoint(pn);

    iftAdjRel *A = iftCircular(1.0);

    double dx, dy;

    if ((initialVoxel.x == finalVoxel.x) && (initialVoxel.y == finalVoxel.y) && (initialVoxel.z == finalVoxel.z)) n = 1;
    else{
        double Dx = pn.x - p1.x;
        double Dy = pn.y - p1.y;

        if (fabs(Dx) >= fabs(Dy)){
            n = (int)(fabs(Dx) + 1);
            dx = iftSign(Dx);
            dy = dx * ((double)Dy/Dx);
        }
        else{
            n = (int)(fabs(Dy) + 1);
            dy = iftSign(Dy);
            dx = dy * ((double)Dx/Dy);
        }
    }


    // Draw first voxel
    // int firstIndex = iftGetVoxelIndex(img, initialVoxel);
    // img->val[firstIndex] = K2;

    drawPoint(img, initialVoxel, A);

    iftPoint p;
    iftVoxel u;

    // iftColor RGB, YCbCr;

    // RGB.val[0] = K2;
    // RGB.val[1] = K2;
    // RGB.val[2] = K2;


    // YCbCr = iftRGBtoYCbCr(RGB, K2);
    p.x = p1.x; p.y = p1.y; p.z = p1.z;

    for (int k = 1; k < n; k++){
        u.x = iftRound(p.x); u.y = iftRound(p.y); u.z = iftRound(p.z);
        drawPoint(img, u, A);
        // if (iftValidVoxel(img, u)) iftDrawPoint(img, u, YCbCr, A, K2);

        p.x += dx;
        p.y += dy;
    }

    iftDestroyAdjRel(&A);
}

double innerProduct(iftVector v1, iftVector v2){
    double result = (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
    return result;
}





int main(int argc, char *argv[]){
    timer *tstart = NULL;
    int    MemDinInicial, MemDinFinal;

    MemDinInicial = iftMemoryUsed(1);

    if ((argc != 7)){
        printf("usage curvilinear_cut: <P1> <P2> <P3> <P4>\n");
        printf("P1: input grayscale 3D image\n");
        printf("P2: segmentation mask\n");
        printf("P3: tilt angle alpha\n");
        printf("P4: sping angle beta\n");
        printf("P5: depth of the cut in mm\n");
        printf("P6: output curvilinear cut rendition\n");
        exit(0);
    }
    tstart = iftTic();

    char filename[200];




    iftImage *img  = iftReadImageByExt(argv[1]);
    iftImage *mask = iftReadImageByExt(argv[2]);
    float alpha    = atof(argv[3]);
    float beta     = atof(argv[4]);
    float depth    = atof(argv[5]);


    /*--------------------------- Envelop ---------------------------*/

    iftImage *bin_mask = binMultipleObjectMask(mask);
    iftDestroyImage(&mask);
    mask               = bin_mask;

    iftImage *envelop = getEnvelopFromSegmentationMask(mask, CLOSE_RADIUS);

    printf("Max: %d, min: %d\n", iftMaximumValue(envelop), iftMinimumValue(envelop));
    iftWriteImageByExt(envelop, "envelop.scn");


    /*--------------------------- EDT ---------------------------*/

    iftImage *EDT = getEDTFromEnvelop(envelop);

    iftWriteImageByExt(EDT, "edt.scn");
    /* Fixing EDT to be -1 outside object */
    for (int p = 0; p < EDT->n; p++){
        if (mask->val[p] == 0) EDT->val[p] = -1;
    }

    /*--------------------------- Curvilinear cut ---------------------------*/



    int diagonal   = (int)sqrt(img->xsize * img->xsize + img->ysize * img->ysize + img->zsize * img->zsize);
    iftImage *cut  = iftCreateImage(diagonal, diagonal, 1);

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


    for (int i = 0; i < cut->n; i++){
        iftVoxel u  = iftGetVoxelCoord(cut, i);
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

            
            iftVoxel intersection;
            intersection.x = iftRound(p0.x + lambda * n_prime.x);
            intersection.y = iftRound(p0.y + lambda * n_prime.y);
            intersection.z = iftRound(p0.z + lambda * n_prime.z);

            if (iftValidVoxel(img, intersection) && lambda > 0){
                if (lambda < lambdaMin) lambdaMin = lambda;
                if (lambda > lambdaMax) lambdaMax = lambda;
            }
        }

        if ((lambdaMin != IFT_INFINITY_DBL) && (lambdaMax != IFT_INFINITY_DBL_NEG)){
            iftPoint initialPoint = createPoint((p0.x + lambdaMin * n_prime.x), (p0.y + lambdaMin * n_prime.y), (p0.z + lambdaMin * n_prime.z));
            iftPoint finalPoint   = createPoint((p0.x + lambdaMax * n_prime.x), (p0.y + lambdaMax * n_prime.y), (p0.z + lambdaMax * n_prime.z));


            iftPoint surfacePoint = findIsoSurfacePoint(EDT, depth, initialPoint, finalPoint);

            if (surfacePoint.x != -10){
                cut->val[i] = trilinearInterpolation(img, surfacePoint);
            }

        }
    }

    radiometricEnhance(cut, cut, 1 - CONTRAST, 1 - BRIGHTNESS);

    /* ----------------------------- WIREFRAME ------------------------*/

    // iftVector n_neg     = createVector(0, 0, -1);
    // centerTranslation   = createVector(-img->xsize / 2, -img->ysize / 2, -img->zsize / 2);
    // diagonalTranslation = createVector(diagonal / 2, diagonal / 2, diagonal / 2);

    // Rx    = iftRotationMatrix(IFT_AXIS_X, alpha);
    // Ry    = iftRotationMatrix(IFT_AXIS_Y, beta);
    // Tc    = iftTranslationMatrix(centerTranslation);
    // Td    = iftTranslationMatrix(diagonalTranslation);

    // Phi_r              = iftMultMatrices(Ry, Rx);
    // aux                = iftMultMatrices(Td, Phi_r);
    // iftMatrix *Phi     = iftMultMatrices(aux, Tc);

    // Vertex *sceneVertex = createSceneVertex(img);
    // Edge *sceneEdges    = createSceneEdges(sceneVertex);
    // for (int f = 0; f < N_SCENE_FACES; f++){
    //     iftVector normalVector = sceneFaces[f].normal_vector;
    //     iftPoint faceCenter    = sceneFaces[f].center;

        
    //     if (innerProduct(iftTransformVector(Phi_r, normalVector), n_neg) > IFT_EPSILON){
    //         for (int e = 0; e < N_SCENE_EDGES; e++){
    //             iftVector edgeVector1, edgeVector2;
    //             edgeVector1 = subtractPointsAsVector(sceneEdges[e].v1.vertex, faceCenter);
    //             edgeVector2 = subtractPointsAsVector(sceneEdges[e].v2.vertex, faceCenter);

    //             if ((innerProduct(edgeVector1, normalVector) == 0) && (innerProduct(edgeVector2, normalVector) == 0)){
    //                 iftPoint p1 = iftTransformPoint(Phi, sceneEdges[e].v1.vertex);
    //                 iftPoint pn = iftTransformPoint(Phi, sceneEdges[e].v2.vertex);

    //                 p1.z = 0;
    //                 pn.z = 0;


    //                 draw2DLine(cut, p1, pn);
    //             }

    //         }
    //     }
        
    // }



    sprintf(filename, "%s", argv[6]);
    iftWriteImageByExt(cut, filename);






    iftDestroyImage(&img);
    iftDestroyImage(&envelop);
    iftDestroyImage(&EDT);
    iftDestroyImage(&mask);
    iftDestroyImage(&cut);

        
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
        
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    
    return(0);
}