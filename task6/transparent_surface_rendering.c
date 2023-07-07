#include <stdio.h>
#include <stdlib.h>

#include "ift.h"

#define CLOSE_RADIUS 20
#define N_SCENE_FACES 6
#define N_SCENE_VERTEX 8
#define N_SCENE_EDGES 12
#define K1 0
#define K2 65535
#define VISIBILITY_EPSILON 0.1
#define RHO 2

#define BRIGHTNESS 0.2
#define CONTRAST 0.1
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

double innerProduct(iftVector v1, iftVector v2){
    double result = (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
    return result;
}
double angleBetweenVectors(iftVector v1, iftVector v2){
    double theta = acos(innerProduct(v1, v2) / (iftVectorMagnitude(v1) * iftVectorMagnitude(v2)));
    return theta;
}

iftVector calculateNormalVector(iftImage *scene, iftImage *label, float alpha, int p){
    /* Calculating precise normal vector */
    iftAdjRel *A = iftSpheric((3.0));
    int       gradient;
    iftVector gradient_vector = createVector(0, 0, 0);
    iftVoxel u = iftGetVoxelCoord(scene, p);
    for (int i = 1; i < A->n; i++){
        iftVoxel v = iftGetAdjacentVoxel(A, u, i);
        if (iftValidVoxel(scene, v)){
            int q = iftGetVoxelIndex(scene, v);

            iftVector adjacent_vector;
            adjacent_vector.x = v.x - u.x;
            adjacent_vector.y = v.y - u.y;
            adjacent_vector.z = v.z - u.z;

            if (label->val[q] == label->val[p]){
                gradient = scene->val[q] - scene->val[p];
            }
            else {
                gradient = 0 - scene->val[p];
            }

            gradient_vector.x += (adjacent_vector.x)/iftVectorMagnitude(adjacent_vector) * gradient;
            gradient_vector.y += (adjacent_vector.y)/iftVectorMagnitude(adjacent_vector) * gradient;
            gradient_vector.z += (adjacent_vector.z)/iftVectorMagnitude(adjacent_vector) * gradient;
        }
    }

    double gradientVectorMagnitude = iftVectorMagnitude(gradient_vector);


    gradient_vector.x = gradient_vector.x / gradientVectorMagnitude;
    gradient_vector.y = gradient_vector.y / gradientVectorMagnitude;
    gradient_vector.z = gradient_vector.z / gradientVectorMagnitude;


    iftVoxel k;
    k.x = u.x + iftRound(alpha * gradient_vector.x);
    k.y = u.y + iftRound(alpha * gradient_vector.y);
    k.z = u.z + iftRound(alpha * gradient_vector.z);


    int l = iftGetVoxelIndex(scene, k);
    int signal = 1;

    if (iftValidVoxel(scene, k)){
        if (label->val[l] != label->val[p]) signal = -1;
    }
    iftVector normalVector;
    
    normalVector.x = signal * gradient_vector.x;
    normalVector.y = signal * gradient_vector.y;
    normalVector.z = signal * gradient_vector.z;

    iftDestroyAdjRel(&A);
    return normalVector;

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


typedef struct _object{
    int label;
    int visibility;
    float opacity;
    float r, g, b;
    iftImage *object_sEDT;
} Object;

typedef struct _normal_context{
    iftVector *normal_vectors;
    iftImage  *normal_idx;
} NormalContext;


int squaredEuclideanDistance(iftVoxel u, iftVoxel v){
    int distance = (int)(((u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y) + (u.z - v.z) * (u.z - v.z)));
    return distance;
}

iftVector *getNormalVectorTable(){
    iftVector *normal_vectors = (iftVector *)calloc(360 * 180, sizeof(iftVector));
    int counter = 0;
    for (int a = 0; a < 360; a++){
        for (int b = -90; b < 90; b++){
            double rad_a = (double)(a) * IFT_PI / 180.0;
            double rad_b = (double)(b) * IFT_PI / 180.0;

            normal_vectors[counter].x = cos(rad_a) * cos(rad_b);
            normal_vectors[counter].y = sin(rad_a) * cos(rad_b);
            normal_vectors[counter].z = sin(rad_b);
            counter++;
        }
    }
    return normal_vectors;
}

int getClosestNormalIndex(iftVector normal_vector, iftVector *normal_table, int n){
    int closest_index        = 0;
    double max_inner_product = IFT_INFINITY_DBL_NEG;
    for (int i = 0; i < n; i++){
        double inner_product = innerProduct(normal_vector, normal_table[i]);
        if (inner_product > max_inner_product){
            closest_index     = i;
            max_inner_product = inner_product;
        }
    }

    return closest_index;
}

NormalContext *getNormalContext(iftImage *scene, iftImage *label, float alpha){
    NormalContext *normal_context = (NormalContext *)calloc(1, sizeof(NormalContext));
    iftImage  *normal_idx      = iftCreateImage(scene->xsize, scene->ysize, scene->zsize);
    iftVector *normal_table    = getNormalVectorTable();
    iftAdjRel *B = iftSpheric(1.0);

    /* Get border voxels */
    iftSet *S    = iftObjectBorderSet(label, B);

    /* Calculate normal vector for every voxel in border set*/
    while(S != NULL){
        int p = iftRemoveSet(&S);
        iftVector normalVector  = calculateNormalVector(scene, label, alpha, p);
        normal_idx->val[p]      = getClosestNormalIndex(normalVector, normal_table, 360 * 180);
    }

    iftDestroyAdjRel(&B);
    iftDestroySet(&S);

    normal_context->normal_idx     = normal_idx;
    normal_context->normal_vectors = normal_table;

    return normal_context;
}

iftImage *sEDT(iftImage *label, int object){

    /* Extracting boundary voxels */
    iftSet *S    = NULL;
    iftAdjRel *B = iftSpheric(1.0);
    for (int p = 0; p < label->n; p++){
        iftVoxel u = iftGetVoxelCoord(label, p);
        if (label->val[p] == object){
            for (int i = 0 ; i < B->n; i++){
                iftVoxel v = iftGetAdjacentVoxel(B, u, i);
                if (iftValidVoxel(label, v)){
                    int q = iftGetVoxelIndex(label, v);
                    if (label->val[q] != label->val[p]){
                        iftInsertSet(&S, p);
                        break;
                    }
                }
                else{
                    iftInsertSet(&S, p);
                    break;
                }
            }
        } 
    }

    iftDestroyAdjRel(&B);

    

    iftImage *cost = iftCreateImageFromImage(label);
    iftImage *root = iftCreateImageFromImage(label);

    for (int p = 0; p < cost->n; p++) {
        cost->val[p] = IFT_INFINITY_INT; 
        root->val[p] = 0;
    }
    
    float diagonal = (label->xsize) * (label->xsize) + (label->ysize) * (label->ysize) + (label->zsize) * (label->zsize);
    iftGQueue *Q   = iftCreateGQueue(diagonal, label->n, cost->val);
    iftAdjRel *A   = iftSpheric(sqrt(3.0));

    while (S != NULL){
        int r = iftRemoveSet(&S);
        cost->val[r] = 0;
        root->val[r] = r;

        iftInsertGQueue(&Q, r);
    }

    while (!iftEmptyGQueue(Q)){
        int p               = iftRemoveGQueue(Q);
        iftVoxel u          = iftGetVoxelCoord(cost, p);
        iftVoxel root_voxel = iftGetVoxelCoord(cost, root->val[p]);
        if (sqrt(cost->val[p]) <= RHO){
            for (int i = 0; i < A->n; i++){
                iftVoxel v = iftGetAdjacentVoxel(A, u, i);
                if (iftValidVoxel(cost, v)){
                    int q = iftGetVoxelIndex(cost, v);
                    if (cost->val[q] > cost->val[p]){
                        double tmp = squaredEuclideanDistance(root_voxel, v);
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
    }



    for (int p = 0; p < cost->n; p++){
        if (cost->val[p] == IFT_INFINITY_INT){ 
            cost->val[p] = 0;
        }
        else if (label->val[p] == object){
            cost->val[p] = -cost->val[p];
        }
    }


    iftDestroyImage(&root);
    iftDestroyAdjRel(&A);
    iftDestroyGQueue(&Q);
    iftDestroySet(&S);

    return cost;

}


Object *createObjects(int n, int visibilities[], float opacities[], iftImage *label){
    Object *objects = (Object *)calloc(n, sizeof(Object));

    objects[0].r = 255 / 255.0;
    objects[0].g = 152 / 255.0;
    objects[0].b = 0 / 255.0;

    objects[1].r = 31 / 255.0;
    objects[1].g = 137 / 255.0;
    objects[1].b = 237 / 255.0;

    // objects[2].r = 197 / 255.0;
    // objects[2].g = 54 / 255.0;
    // objects[2].b = 237 / 255.0;
    

    for (int i = 0; i < n; i++){
        objects[i].visibility = visibilities[i];
        objects[i].opacity    = opacities[i];
        objects[i].object_sEDT = sEDT(label, i + 1);
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

void getMinMaxSceneDistances(iftImage *label, float alpha, float beta, float *dmin, float *dmax){
    int diagonal   = (int)sqrt(label->xsize * label->xsize + label->ysize * label->ysize + label->zsize * label->zsize);

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

    while (S != NULL){
        int p       = iftRemoveSet(&S);
        iftVoxel u  = iftGetVoxelCoord(label, p);
        iftPoint q  = createPoint(u.x, u.y, u.z);
        iftPoint p0 = iftTransformPoint(Phi, q);

        float d = sqrt((p0.x - diagonal / 2) * (p0.x - diagonal / 2) + (p0.y - diagonal / 2) * (p0.y - diagonal / 2) + (p0.z + diagonal / 2) * (p0.z + diagonal / 2));
        if (d > *dmax) *dmax = d;
        if (d < *dmin) *dmin = d;
    }

    iftDestroyAdjRel(&A);
    iftDestroySet(&S);
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
    int   ns, ra;
    float dmax, dmin;

    // Objects informations
    int n_objects;
    Object *objects;
    iftImage *label;

    // Normal vectors
    NormalContext *normal_context;

} GraphicalContext;
int findSurfacePoint(iftImage *scene, GraphicalContext *gc, iftPoint p1, iftPoint pn, double *theta, iftVector n_prime){

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
    iftAdjRel *A     = iftSpheric(1.0);

    for (int i = 0; i < A->n; i++){
        iftVoxel v = iftGetAdjacentVoxel(A, initialVoxel, i);
        if (iftValidVoxel(gc->label, v)){
            int q = iftGetVoxelIndex(gc->label, v);
            if ((gc->label->val[q] != 0) && (gc->objects[gc->label->val[q] - 1].visibility)){
                iftVector normal_vector = calculateNormalVector(scene, gc->label, 2.0, q);
                double tmp_theta        = angleBetweenVectors(normal_vector, n_prime);
                if ((cos(tmp_theta) > 0)){
                    iftDestroyAdjRel(&A);
                    *theta = tmp_theta;
                    return q;
                }
            }
        }
    }
    
    for (int k = 1; k < n; k++){
        p_prime.x += dx;
        p_prime.y += dy;
        p_prime.z += dz;
        
        iftVoxel p_prime_voxel = getVoxelFromPoint(p_prime);
        for (int i = 0; i < A->n; i++){
            iftVoxel v = iftGetAdjacentVoxel(A, p_prime_voxel, i);
            if (iftValidVoxel(gc->label, v)){
                int q = iftGetVoxelIndex(gc->label, v);
                if ((gc->label->val[q] != 0) && (gc->objects[gc->label->val[q] - 1].visibility)){
                    iftVector normal_vector = calculateNormalVector(scene, gc->label, 2.0, q);
                    double tmp_theta        = angleBetweenVectors(normal_vector, n_prime);
                    if ((cos(tmp_theta) > 0)){
                        iftDestroyAdjRel(&A);
                        *theta = tmp_theta;
                        return q;
                    }
                }
            }
        }
    }

    iftDestroyAdjRel(&A);

    return IFT_NIL;

}

GraphicalContext *createGraphicalContext(iftImage *scene, iftImage *label, int n_objects, float alpha, float beta, int visibilities[], float opacities[]){
    GraphicalContext *gc = (GraphicalContext *)calloc(1, sizeof(GraphicalContext));

    gc->ka = 0.1;
    gc->ks = 0.2;
    gc->kd = 0.7;
    gc->ns = 5;
    gc->ra = K2;
    gc->n_objects      = n_objects;
    gc->objects        = createObjects(gc->n_objects, visibilities, opacities, label);
    gc->label          = label;
    gc->dmin           = IFT_INFINITY_FLT;
    gc->dmax           = IFT_INFINITY_FLT_NEG;
    getMinMaxSceneDistances(gc->label, alpha, beta, &(gc->dmin), &(gc->dmax));

    gc->normal_context = NULL;
    // gc->normal_context = getNormalContext(scene, label, 2);

    return gc;
}

void destroyGraphicalContext(GraphicalContext **gc){
    if ((*gc) != NULL){
        free((*gc)->objects);
        if ((*gc)->normal_context != NULL){
            iftDestroyImage(&((*gc)->normal_context->normal_idx));
            free((*gc)->normal_context->normal_vectors);
        }

        for (int i = 0; i < (*gc)->n_objects; i++){
            if ((*gc)->objects[i].object_sEDT != NULL) iftDestroyImage(&((*gc)->objects[i].object_sEDT));
        }
        free((*gc));
        (*gc) = NULL;
    }
}


iftColor phongModel(iftImage *scene, GraphicalContext *gc, int surfacePoint, iftPoint planePoint, double theta){
    iftColor RGB;

    double ambient = 0, diffuse = 0, specular = 0;
    /* Ambient component */
    ambient = gc->ka * gc->ra;

    /* Diffuse component */
    if (cos(theta) > 0)
        diffuse = gc->kd * cos(theta);

    /* Specular component */
    if (cos(2.0 * theta) > 0)
        specular = gc->ks * pow(cos(2 * theta), gc->ns);


    /* Depth shading */
    iftVoxel surfaceVoxel = iftGetVoxelCoord(gc->label, surfacePoint);
    double point_distance = sqrt((surfaceVoxel.x - planePoint.x) * (surfaceVoxel.x - planePoint.x) + (surfaceVoxel.y - planePoint.y) * (surfaceVoxel.y - planePoint.y) + (surfaceVoxel.z - planePoint.z) * (surfaceVoxel.z - planePoint.z));    
    double depth_shading  = K2 * (1 - (point_distance - gc->dmin) / (gc->dmax - gc ->dmin));

    /* Phong's reflected light*/
    double reflected_light = ambient + depth_shading * (diffuse + specular);

    /* Object final color */
    int object_index = gc->label->val[surfacePoint] - 1;
    RGB.val[0]       = gc->objects[object_index].visibility * (reflected_light * gc->objects[object_index].r);
    RGB.val[1]       = gc->objects[object_index].visibility * (reflected_light * gc->objects[object_index].g);
    RGB.val[2]       = gc->objects[object_index].visibility * (reflected_light * gc->objects[object_index].b);


    // return reflected_light;
    return iftRGBtoYCbCr(RGB, K2);
}

int IntphongModel(iftImage *scene, GraphicalContext *gc, int surfacePoint, iftPoint planePoint, double theta){
    double ambient = 0, diffuse = 0, specular = 0;

    /* Ambient component */
    ambient = gc->ka * gc->ra;

    /* Diffuse component */
    if (cos(theta) > 0)
        diffuse = gc->kd * cos(theta);

    /* Specular component */
    if (cos(2.0 * theta) > 0)
        specular = gc->ks * pow(cos(2 * theta), gc->ns);


    /* Depth shading */
    iftVoxel surfaceVoxel = iftGetVoxelCoord(gc->label, surfacePoint);
    double point_distance = sqrt((surfaceVoxel.x - planePoint.x) * (surfaceVoxel.x - planePoint.x) + (surfaceVoxel.y - planePoint.y) * (surfaceVoxel.y - planePoint.y) + (surfaceVoxel.z - planePoint.z) * (surfaceVoxel.z - planePoint.z));    
    double depth_shading  = K2 * (1 - (point_distance - gc->dmin) / (gc->dmax - gc ->dmin));

    /* Phong's reflected light*/
    int object_index       = gc->label->val[surfacePoint] - 1;
    double reflected_light = gc->objects[object_index].visibility * (ambient + depth_shading * (diffuse + specular));

    // return reflected_light;
    return reflected_light;
}

int countNumberOfObjectsInLabelScene(iftImage *label){
    int counter = 0;
    for (int p = 0; p < label->n; p++){
        if (label->val[p] > counter) counter = label->val[p];
    }

    return counter;
}

iftImage *getLabeledSagittalSlice(iftImage *img, int x, int perspective, GraphicalContext *gc){
    iftImage *sagittalSlice = iftCreateImage(img->ysize, img->zsize, 1);
    iftSetCbCr(sagittalSlice, K2 / 2);


    iftVoxel u;
    iftColor RGB, Y_CB_CR;

    int k = 0;
    // radiologist
    if (perspective == 0){         u.x = x;
        for (u.z = img->zsize - 1; u.z >= 0; u.z--){
            for (u.y = 0; u.y < img->ysize; u.y++){ 
                int p = iftGetVoxelIndex(img, u);
                sagittalSlice->val[k] = img->val[p];

                if (gc->label->val[p]){
                    // sagittalSlice->val[k] = (int)((0.6 * K2) * img->val[p] + (0.4 * K2) * gc->objects[gc->label->val[p] - 1].object_sEDT->val[p]);
                    RGB.val[0]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].r);
                    RGB.val[1]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].g);
                    RGB.val[2]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].b);
                    Y_CB_CR               = iftRGBtoYCbCr(RGB, K2);
                    sagittalSlice->Cb[k]   = Y_CB_CR.val[1];
                    sagittalSlice->Cr[k]   = Y_CB_CR.val[2];
                    if (gc->objects[gc->label->val[p] - 1].object_sEDT->val[p] == 0){
                        RGB.val[0] = K2;
                        RGB.val[1] = 0;
                        RGB.val[2] = 0;
                        Y_CB_CR               = iftRGBtoYCbCr(RGB, K2);
                        // sagittalSlice->val[k]  = Y_CB_CR.val[0];
                        sagittalSlice->Cb[k]   = Y_CB_CR.val[1];
                        sagittalSlice->Cr[k]   = Y_CB_CR.val[2];

                }
                }
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
                if (gc->label->val[p]){
                    RGB.val[0]            = gc->objects[gc->label->val[p] - 1].r;
                    RGB.val[1]            = gc->objects[gc->label->val[p] - 1].g;
                    RGB.val[2]            = gc->objects[gc->label->val[p] - 1].b;
                    Y_CB_CR               = iftRGBtoYCbCr(RGB, K2);
                    sagittalSlice->Cb[k]  = Y_CB_CR.val[1];
                    sagittalSlice->Cr[k]  = Y_CB_CR.val[2];
                    if (gc->objects[gc->label->val[p] - 1].object_sEDT->val[p] == 0){
                        RGB.val[0] = K2;
                        RGB.val[1] = 0;
                        RGB.val[2] = 0;
                        Y_CB_CR               = iftRGBtoYCbCr(RGB, K2);
                        // sagittalSlice->val[k]  = Y_CB_CR.val[0];
                        sagittalSlice->Cb[k]   = Y_CB_CR.val[1];
                        sagittalSlice->Cr[k]   = Y_CB_CR.val[2];

                }
                }
                k++;
            }
        }
    }
    return sagittalSlice;
}


iftImage *getLabeledCoronalSlice(iftImage *img, int y, int perspective, GraphicalContext *gc){
    iftImage *coronalSlice = iftCreateImage(img->xsize, img->zsize, 1);
    iftSetCbCr(coronalSlice, K2 / 2);


    iftVoxel u;
    iftColor RGB, Y_CB_CR;

    int k = 0;
    // radiologist
    if (perspective == 0){ 
        u.y = y;
        for (u.z = img->zsize - 1; u.z >= 0; u.z--){
            for (u.x = 0; u.x < img->xsize; u.x++){
                int p = iftGetVoxelIndex(img, u);
                coronalSlice->val[k] = img->val[p];
                if (gc->label->val[p]){
                    RGB.val[0]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].r);
                    RGB.val[1]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].g);
                    RGB.val[2]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].b);
                    Y_CB_CR               = iftRGBtoYCbCr(RGB, K2);
                    coronalSlice->Cb[k]   = Y_CB_CR.val[1];
                    coronalSlice->Cr[k]   = Y_CB_CR.val[2];
                    if (gc->objects[gc->label->val[p] - 1].object_sEDT->val[p] == 0){
                        RGB.val[0] = K2;
                        RGB.val[1] = 0;
                        RGB.val[2] = 0;
                        Y_CB_CR               = iftRGBtoYCbCr(RGB, K2);
                        // sagittalSlice->val[k]  = Y_CB_CR.val[0];
                        coronalSlice->Cb[k]   = Y_CB_CR.val[1];
                        coronalSlice->Cr[k]   = Y_CB_CR.val[2];

                }
                }
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
                    if (gc->label->val[p]){
                        RGB.val[0]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].r);
                        RGB.val[1]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].g);
                        RGB.val[2]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].b);
                        Y_CB_CR               = iftRGBtoYCbCr(RGB, K2);
                        coronalSlice->Cb[k]   = Y_CB_CR.val[1];
                        coronalSlice->Cr[k]   = Y_CB_CR.val[2];
                        
                    }
                    k++;
                }
            }
    }
    return coronalSlice;
}


iftImage *getLabeledAxialSlice(iftImage *img, int z, int perspective, GraphicalContext *gc){
    iftImage *axialSlice = iftCreateImage(img->xsize, img->ysize, 1);
    iftSetCbCr(axialSlice, K2 / 2);


    iftVoxel u;
    iftColor RGB, Y_CB_CR;

    int k = 0;
    // radiologist
    if (perspective == 0){
        u.z = z;
        for (u.y = 0; u.y < img->ysize; u.y++){
            for (u.x = 0; u.x < img->xsize; u.x++){
                int p = iftGetVoxelIndex(img, u);
                axialSlice->val[k] = img->val[p];
                if (gc->label->val[p]){
                    RGB.val[0]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].r);
                    RGB.val[1]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].g);
                    RGB.val[2]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].b);
                    Y_CB_CR               = iftRGBtoYCbCr(RGB, K2);
                    axialSlice->Cb[k]   = Y_CB_CR.val[1];
                    axialSlice->Cr[k]   = Y_CB_CR.val[2];
                    if (gc->objects[gc->label->val[p] - 1].object_sEDT->val[p] == 0){
                        RGB.val[0] = K2;
                        RGB.val[1] = 0;
                        RGB.val[2] = 0;
                        Y_CB_CR               = iftRGBtoYCbCr(RGB, K2);
                        // sagittalSlice->val[k]  = Y_CB_CR.val[0];
                        axialSlice->Cb[k]   = Y_CB_CR.val[1];
                        axialSlice->Cr[k]   = Y_CB_CR.val[2];

                }
                }
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
                if (gc->label->val[p]){
                    RGB.val[0]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].r);
                    RGB.val[1]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].g);
                    RGB.val[2]            = (int)(K2 * gc->objects[gc->label->val[p] - 1].b);
                    Y_CB_CR               = iftRGBtoYCbCr(RGB, K2);
                    axialSlice->Cb[k]   = Y_CB_CR.val[1];
                    axialSlice->Cr[k]   = Y_CB_CR.val[2];
                }
                k++;
            }
        }
    }
    return axialSlice;
}



iftVector calculateNormalVectorBySEDT(iftImage *objectSEDT, int p){
    iftAdjRel *A = iftSpheric((3.0));
    int       gradient;
    iftVector gradient_vector = createVector(0, 0, 0);
    iftVoxel u                = iftGetVoxelCoord(objectSEDT, p);

    for (int i = 1; i < A->n; i++){
        iftVoxel v = iftGetAdjacentVoxel(A, u, i);
        if (iftValidVoxel(objectSEDT, v)){
            int q = iftGetVoxelIndex(objectSEDT, v);

            iftVector adjacent_vector;
            adjacent_vector.x = v.x - u.x;
            adjacent_vector.y = v.y - u.y;
            adjacent_vector.z = v.z - u.z;

            gradient = objectSEDT->val[q] - objectSEDT->val[p];

            // printf("Val p: %d\n", objectSEDT->val[p]);
            // printf("Val q: %d\n", objectSEDT->val[q]);

            // printf("Gradient: %d\n", gradient);
            // printf("u: %d %d %d\n", u.x, u.y, u.z);
            // printf("v: %d %d %d\n", v.x, v.y, v.z);

            // printf("Adjacent vector: %lf %lf %lf -> %f, %d = %d - %d\n", adjacent_vector.x, adjacent_vector.y, adjacent_vector.z, iftVectorMagnitude(adjacent_vector), gradient, objectSEDT->val[q], objectSEDT->val[p]);

            gradient_vector.x += (adjacent_vector.x)/iftVectorMagnitude(adjacent_vector) * gradient;
            gradient_vector.y += (adjacent_vector.y)/iftVectorMagnitude(adjacent_vector) * gradient;
            gradient_vector.z += (adjacent_vector.z)/iftVectorMagnitude(adjacent_vector) * gradient;
        }
    }

    iftDestroyAdjRel(&A);

    // printf("Gradient vector: %lf %lf %lf -> %f\n", gradient_vector.x, gradient_vector.y, gradient_vector.z, iftVectorMagnitude(gradient_vector));
    // printf("##########################\n");

    double gradientVectorMagnitude = iftVectorMagnitude(gradient_vector);
    gradient_vector.x =  - gradient_vector.x / gradientVectorMagnitude;
    gradient_vector.y =  - gradient_vector.y / gradientVectorMagnitude;
    gradient_vector.z =  - gradient_vector.z / gradientVectorMagnitude;

    return gradient_vector;

}

iftColor computeColorAlongRay(iftImage *scene, GraphicalContext *gc, iftPoint p1, iftPoint pn, iftPoint planePoint, iftVector n_prime, int normal_type){
    int n;

    iftVoxel initialVoxel = getVoxelFromPoint(p1);
    iftVoxel finalVoxel   = getVoxelFromPoint(pn);

    double dx = 0, dy = 0, dz = 0;

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

    int *obj_flag                = (int *)calloc(gc->n_objects, sizeof(int));
    double visibility_saturation = 1.0;

    iftColor RGB;
    RGB.val[0] = 0;
    RGB.val[1] = 0;
    RGB.val[2] = 0; 

    int k = 1;



    iftAdjRel *A     = iftSpheric(1.0);
    iftPoint p_prime = createPoint(initialVoxel.x, initialVoxel.y, initialVoxel.z);
    while ((k <= n) && (visibility_saturation > VISIBILITY_EPSILON)){

        iftVoxel p_prime_voxel = getVoxelFromPoint(p_prime);
        for (int i = 0; i < A->n; i++){
            iftVoxel v = iftGetAdjacentVoxel(A, p_prime_voxel, i);
            if (iftValidVoxel(scene, v)){
                int q = iftGetVoxelIndex(scene, v);
                if ((gc->label->val[q] != 0)){
                    int object_index = gc->label->val[q] - 1;
                    if ((gc->objects[object_index].visibility) && (obj_flag[object_index] == 0 && (gc->objects[object_index].opacity > 0))){
                        iftVector normal_vector;
                        if (normal_type == 0)
                            normal_vector = calculateNormalVector(scene, gc->label, 2.0, q);
                        else
                            normal_vector = calculateNormalVectorBySEDT(gc->objects[object_index].object_sEDT, q);
                        double theta  = angleBetweenVectors(normal_vector, n_prime);
                        if (cos(theta) > 0){
                            /* We found a voxel that must be rendered. Time to compute phong */
                            double reflected_light = IntphongModel(scene, gc, q, planePoint, theta);
                            RGB.val[0] += visibility_saturation * gc->objects[object_index].opacity * reflected_light * gc->objects[object_index].r;
                            RGB.val[1] += visibility_saturation * gc->objects[object_index].opacity * reflected_light * gc->objects[object_index].g;
                            RGB.val[2] += visibility_saturation * gc->objects[object_index].opacity * reflected_light * gc->objects[object_index].b;

                            visibility_saturation *= (1 - gc->objects[object_index].opacity);
                            obj_flag[object_index] = 1;
                            break;
                        }

                    }

                }
            }

        }
        
        p_prime.x += dx;
        p_prime.y += dy;
        p_prime.z += dz;
        k++;
    
    }

    iftDestroyAdjRel(&A);
    free(obj_flag);

    return iftRGBtoYCbCr(RGB, K2);

}


int main(int argc, char *argv[]){
    timer *tstart = NULL;
    int    MemDinInicial, MemDinFinal;

    MemDinInicial = iftMemoryUsed(1);

    if ((argc != 9 && argc != 10)){
        printf("argc: %d\n", argc);
        printf("usage transparent_surface_rendering: <P1> <P2> <P3> <P4> <P5> <P6> <P7> <P8> <P9> <P10> <P11>\n");
        printf("P1: input grayscale 3D image (.scn)\n");
        printf("P2: label scene (.scn)\n");
        printf("P3: tilt angle alpha\n");
        printf("P4: sping angle beta\n");
        printf("P5: output surface rendering\n");
        printf("P6: for scene-based normal vector and 1 for object-based normal vector\n");
        printf("P7: opacities of the objects in P2 (e.g., “o1, o2, o3” for three objects in P2)\n");
        printf("P8: visibility of each object (e.g., “0,1,1” for three objects means that only objects o2 and o3 are visible).\n");
        printf("P9: optional p0 = (x0, y0, z0) coordinates of a point p0 in the scene\n");

        exit(0);
    }
    tstart = iftTic();

    char filename[200];

    iftImage *img       = iftReadImageByExt(argv[1]);
    iftImage *label     = iftReadImageByExt(argv[2]);
    float alpha         = atof(argv[3]);
    float beta          = atof(argv[4]);
    int normal_method   = atoi(argv[6]);
    float o1, o2, o3;
    int   v1, v2, v3;
    int x0 = 0, y0 = 0, z0 = 0;
    sscanf(argv[7], "%f %f %f", &o1, &o2, &o3);
    sscanf(argv[8], "%d %d %d", &v1, &v2, &v3);
    if (argc == 10)
        sscanf(argv[9], "%d %d %d", &x0, &y0, &z0);
    


    float opacities[3]  = { o1, o2, o3 };
    int visibilities[3] = { v1, v2, v3 };

    

    int n_objects        = countNumberOfObjectsInLabelScene(label);
    GraphicalContext *gc = createGraphicalContext(img, label, n_objects, alpha, beta, visibilities, opacities);
    int diagonal         = (int)sqrt(img->xsize * img->xsize + img->ysize * img->ysize + img->zsize * img->zsize);
    iftImage *surface    = iftCreateImage(diagonal, diagonal, 1);
    iftSetCbCr(surface, K2 / 2);

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

    
    iftVector n_prime = iftTransformVector(Phi_r, n);
    Face *sceneFaces  = createSceneFaces(img);

    
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

            iftColor c = computeColorAlongRay(img, gc, initialPoint, finalPoint, p0, n_prime, normal_method);
            surface->val[i] = c.val[0];
            surface->Cb[i]  = c.val[1];
            surface->Cr[i]  = c.val[2];
        }
    }

    iftImage *aux_image = iftNormalize(surface, 0, K2);
    iftDestroyImage(&surface);
    surface = aux_image;


    /* Getting slices overlaid by transparent colors */
    iftImage *sagital = getLabeledSagittalSlice(img, x0, 0, gc);
    iftImage *coronal = getLabeledCoronalSlice(img,  y0, 0, gc);
    iftImage *axial   = getLabeledAxialSlice(img,    z0, 0, gc);


    radiometricEnhance(img, sagital, 1.0 - CONTRAST, 1.0 - BRIGHTNESS);
    radiometricEnhance(img, coronal, 1.0 - CONTRAST, 1.0 - BRIGHTNESS);
    radiometricEnhance(img, axial,   1.0 - CONTRAST, 1.0 - BRIGHTNESS);
    

    // radiometricEnhance(surface, surface, 1 - CONTRAST, 1 - BRIGHTNESS);


    sprintf(filename, "%s", argv[5]);
    iftWriteImageByExt(surface, filename);

    sprintf(filename, "sagital_%s", argv[5]);
    iftWriteImageByExt(sagital, filename);

    sprintf(filename, "coronal_%s", argv[5]);
    iftWriteImageByExt(coronal, filename);

    sprintf(filename, "axial_%s", argv[5]);
    iftWriteImageByExt(axial, filename);

    iftDestroyImage(&img);
    iftDestroyImage(&label);
    iftDestroyImage(&surface);
    iftDestroyImage(&sagital);
    iftDestroyImage(&coronal);
    iftDestroyImage(&axial);


    iftDestroyMatrix(&Rx);
    iftDestroyMatrix(&Ry);
    iftDestroyMatrix(&Tc);
    iftDestroyMatrix(&Td);
    iftDestroyMatrix(&Phi_r);
    iftDestroyMatrix(&aux);
    iftDestroyMatrix(&Phi_inv);
    destroyGraphicalContext(&gc);
    free(sceneFaces);

        
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
        
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    
    return(0);
}