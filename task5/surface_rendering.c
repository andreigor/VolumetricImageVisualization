#include <stdio.h>
#include <stdlib.h>

#include "ift.h"

#define CLOSE_RADIUS 20
#define N_SCENE_FACES 6
#define N_SCENE_VERTEX 8
#define N_SCENE_EDGES 12
#define K1 0
#define K2 4095

#define BRIGHTNESS 0.3
#define CONTRAST 0.3
// #define DEBUG 1


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

iftPoint findSurfacePoint(iftImage *label, iftPoint p1, iftPoint pn){

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
    

    iftPoint p_prime = createPoint(initialVoxel.x, initialVoxel.y, initialVoxel.z);
    int p_prime_index = iftGetVoxelIndex(label, initialVoxel);

    if (label->val[p_prime_index] != 0)
        return p_prime;
    
    for (int k = 1; k < n; k++){
        p_prime.x += dx;
        p_prime.y += dy;
        p_prime.z += dz;
        
        iftVoxel p_prime_voxel = getVoxelFromPoint(p_prime);
        // p_prime_voxel.x = iftRound(p_prime.x); p_prime_voxel.y = iftRound(p_prime.y); p_prime_voxel.z = iftRound(p_prime.z);
        p_prime_index = iftGetVoxelIndex(label, p_prime_voxel);
        if (label->val[p_prime_index] != 0)
            return p_prime;
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

typedef struct _object{
    int label;
    float r, g, b;
} Object;

typedef struct _normal_context{
    iftVector *normal_vectors;
    iftImage  *normal_idx;
} NormalContext;


int squaredEuclideanDistance(iftVoxel u, iftVoxel v){
    int distance = (int)(((u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y) + (u.z - v.z) * (u.z - v.z)));
    return distance;
}

NormalContext getNormalContext(iftImage *scene, iftImage *label, float alpha){
    iftImage *normal_idx      = iftCreateImage(scene->xsize, scene->ysize, scene->zsize);
    iftVector *normal_vectors = (iftVector *)calloc(scene->n / 2, sizeof(iftVector));
    NormalContext normal_context;

    iftAdjRel *A = iftSpheric(sqrt(3.0));

    int idx_counter = 1;
    for (int p = 0; p < scene->n; p++){
        normal_idx->val[p] = 0;
        if (label->val[p]){
            float     gradient;
            iftVector gradient_vector = createVector(0, 0, 0);
            iftVoxel u = iftGetVoxelCoord(scene, p);
            for (int i = 0; i < A->n; i++){
                iftVoxel v = iftGetAdjacentVoxel(A, u, i);
                if (iftValidVoxel(scene, v)){
                    int q = iftGetVoxelIndex(scene, v);

                    iftVector adjacent_vector;
                    adjacent_vector.x = v.x - u.x;
                    adjacent_vector.y = v.y - u.y;
                    adjacent_vector.z = v.z - u.z; 
                    gradient = scene->val[q] - scene->val[p];
                    

                    gradient_vector.x += (adjacent_vector.x)/iftVectorMagnitude(adjacent_vector) * gradient;
                    gradient_vector.y += (adjacent_vector.y)/iftVectorMagnitude(adjacent_vector) * gradient;
                    gradient_vector.z += (adjacent_vector.z)/iftVectorMagnitude(adjacent_vector) * gradient;
                }
            }

            gradient_vector.x = gradient_vector.x / iftVectorMagnitude(gradient_vector);
            gradient_vector.y = gradient_vector.y / iftVectorMagnitude(gradient_vector);
            gradient_vector.z = gradient_vector.z / iftVectorMagnitude(gradient_vector);

            iftVoxel k;
            k.x = u.x + iftRound(alpha * gradient_vector.x);
            k.y = u.y + iftRound(alpha * gradient_vector.y);
            k.z = u.z + iftRound(alpha * gradient_vector.z);

            int l = iftGetVoxelIndex(scene, k);
            int signal = 1;
            if (label->val[l] != label->val[p]) signal = -1;

            iftVector normalVector;
            
            normalVector.x = signal * gradient_vector.x;
            normalVector.y = signal * gradient_vector.y;
            normalVector.z = signal * gradient_vector.z;

            normal_idx->val[p] = idx_counter;
            normal_vectors[idx_counter] = normalVector;
        }

    }
    normal_context.normal_idx = normal_idx;
    normal_context.normal_vectors = normal_vectors;

    return normal_context;
}


Object *createObjects(int n){
    Object *objects = (Object *)calloc(n, sizeof(Object));

    for (int i = 0; i < n; i++){
        float r = (float)(rand())/RAND_MAX;
        float g = (float)(rand())/RAND_MAX;
        float b = (float)(rand())/RAND_MAX;

        objects[i].r = r;
        objects[i].g = g;
        objects[i].b = b;

        objects[i].label = i + 1;
    }
    return objects;
}

iftVector subtractPointsAsVector(iftPoint p1, iftPoint p2){
    iftVector r;
    r.x = p1.x - p2.x;
    r.y = p1.y - p2.y;
    r.z = p1.z - p2.z;

    return r;
}

double innerProduct(iftVector v1, iftVector v2){
    double result = (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
    return result;
}

void getMinMaxSceneDistances(iftImage *label, float alpha, float beta, float *dmin, float *dmax){
    int diagonal   = (int)sqrt(label->xsize * label->xsize + label->ysize * label->ysize + label->zsize * label->zsize);


    // iftVector n_neg               = createVector(0, 0, -1);
    iftVector centerTranslation   = createVector(-label->xsize / 2, -label->ysize / 2, -label->zsize / 2);
    iftVector diagonalTranslation = createVector(diagonal / 2, diagonal / 2, diagonal / 2);

    iftMatrix *Rx    = iftRotationMatrix(IFT_AXIS_X, alpha);
    iftMatrix *Ry    = iftRotationMatrix(IFT_AXIS_Y, beta);
    iftMatrix *Tc    = iftTranslationMatrix(centerTranslation);
    iftMatrix *Td    = iftTranslationMatrix(diagonalTranslation);

    iftMatrix *Phi_r = iftMultMatrices(Ry, Rx);
    iftMatrix *aux   = iftMultMatrices(Td, Phi_r);
    iftMatrix *Phi   = iftMultMatrices(aux, Tc);

    iftAdjRel *A = iftSpheric(1.0);
    iftSet *S    = iftObjectBorderSet(label, A);

    // int i = 0;
    // Face *sceneFaces = createSceneFaces(label);
    while (S != NULL){
        int p       = iftRemoveSet(&S);
        iftVoxel u  = iftGetVoxelCoord(label, p);
        iftPoint q  = createPoint(u.x, u.y, u.z);
        iftPoint p0 = iftTransformPoint(Phi, q);

        float d = p0.z + diagonal/2;
        if (d > *dmax) *dmax = d;
        if (d < *dmin) *dmin = d;


        // checking if voxel belongs to a visible face
        // for (int f = 0; f < N_SCENE_FACES; f++){

        //     iftVector normalVector = sceneFaces[f].normal_vector;
        //     iftPoint  faceCenter   = sceneFaces[f].center;

            
        //     if (innerProduct(iftTransformVector(Phi_r, normalVector), n_neg) > IFT_EPSILON){
        //         // printf("Face visivel: %d\n", i);
        //         iftVector pointVector = subtractPointsAsVector(p0, faceCenter);
        //         if ((innerProduct(pointVector, normalVector) == 0)){
        //             printf("Ponto visivel: %d\n", i);
        //             float d = p0.z + diagonal/2;
        //             if (d > *dmax) *dmax = d;
        //             if (d < *dmin) *dmin = d;
        //             break; // it belongs to a visible face, so no need to check any further
        //         }
        //     }
        // }
    
        // i++;
    }

    // printf("i: %d\n", i);
    iftDestroyMatrix(&Rx);
    iftDestroyMatrix(&Ry);
    iftDestroyMatrix(&Tc);
    iftDestroyMatrix(&Td);
    iftDestroyMatrix(&Phi_r);
    iftDestroyMatrix(&aux);
    iftDestroyMatrix(&Phi);
}

typedef struct _graphical_context{
    // Phong's parameters
    float ka, ks, kd;
    int   ns, ra, dmax, dmin;

    // Objects informations
    int n_objects;
    Object *objects;

    // Normal vectors
    NormalContext normal_context;

} GraphicalContext;

GraphicalContext *createGraphicalContext(iftImage *scene, iftImage *label, int n_objects){
    GraphicalContext *gc = (GraphicalContext *)calloc(1, sizeof(GraphicalContext));

    gc->ka = 0.1;
    gc->ks = 0.2;
    gc->kd = 0.7;
    gc->ns = 5;
    gc->ra = K2;
    gc->n_objects      = n_objects;
    gc->objects        = createObjects(gc->n_objects);
    gc->normal_context = getNormalContext(scene, label, 2);

    return gc;
}

iftColor phongModel(GraphicalContext *gc, iftPoint p, iftImage *label){
    iftVoxel u = getVoxelFromPoint(p);  

    int voxel_index = iftGetVoxelIndex(label, u);
    int objectLabel = label->val[voxel_index];

    
}







int main(int argc, char *argv[]){
    timer *tstart = NULL;
    int    MemDinInicial, MemDinFinal;

    MemDinInicial = iftMemoryUsed(1);

    if ((argc != 6)){
        printf("usage curvilinear_surface: <P1> <P2> <P3> <P4>\n");
        printf("P1: input grayscale 3D image (.scn)\n");
        printf("P2: label scene (.scn)\n");
        printf("P3: tilt angle alpha\n");
        printf("P4: sping angle beta\n");
        printf("P5: output surface rendering\n");
        exit(0);
    }
    tstart = iftTic();

    char filename[200];

    float dmax = IFT_INFINITY_FLT_NEG;
    float dmin = IFT_INFINITY_FLT;



    iftImage *img   = iftReadImageByExt(argv[1]);
    iftImage *label = iftReadImageByExt(argv[2]);
    float alpha     = atof(argv[3]);
    float beta      = atof(argv[4]);

    getMinMaxSceneDistances(label, alpha, beta, &dmin, &dmax);
    printf("dmax, dmin: %f, %f\n", dmax, dmin);

    exit(0);

    int diagonal       = (int)sqrt(img->xsize * img->xsize + img->ysize * img->ysize + img->zsize * img->zsize);
    iftImage *surface  = iftCreateImage(diagonal, diagonal, 1);

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


    for (int i = 0; i < surface->n; i++){
        iftVoxel u  = iftGetVoxelCoord(surface, i);
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


            iftPoint surfacePoint = findSurfacePoint(label, initialPoint, finalPoint);

            if (surfacePoint.x != -10){
                surface->val[i] = trilinearInterpolation(img, surfacePoint);
            }

        }
    }

    radiometricEnhance(surface, surface, 1 - CONTRAST, 1 - BRIGHTNESS);


    sprintf(filename, "%s", argv[6]);
    iftWriteImageByExt(surface, filename);






    iftDestroyImage(&img);
    iftDestroyImage(&label);
    iftDestroyImage(&surface);

        
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
        
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    
    return(0);
}