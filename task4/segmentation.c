#include "ift.h"
#define alpha 0



float featureEuclidianDistance(iftMImage *feature, int p, int q){
    float sum = 0.0;
    for (int i = 0; i < feature->m; i++){
        sum += (feature->val[p][i] - feature->val[q][i]) * (feature->val[p][i] - feature->val[q][i]);
    }

    return sum / feature->m;
}


  
iftImage *SegmentByWatershed(iftImage *img, iftLabeledSet *seeds, iftMImage *feature_vector)
{
  iftImage   *pathval = NULL, *label = NULL, *gradI=NULL, *gradO = NULL;
  iftGQueue  *Q = NULL;
  int            i, p, q, tmp;
  iftVoxel       u, v;
  iftLabeledSet *S = seeds;
  iftAdjRel     *A = iftSpheric(1.0);
//   printf("Tamanho do feature vector: %ld\n", feature_vector->m);
//   printf("Dimensões da imagem original: %d, %d, %d\n", img->xsize, img->ysize, img->zsize);
//   printf("Dimensões da mimg: %d, %d, %d\n", feature_vector->xsize, feature_vector->ysize, feature_vector->zsize);

  // Initialization

  iftImage *fimg = iftMImageToImage(feature_vector, 4095, 0);
  
  pathval    = iftCreateImage(img->xsize, img->ysize, img->zsize);
  label      = iftCreateImage(img->xsize, img->ysize, img->zsize);
  gradI      = iftImageGradientMagnitude(img,A);
  gradO      = iftImageGradientMagnitude(fimg, A);
//   gradI      = iftCopyImage(img);

  Q          = iftCreateGQueue(powf(iftMaximumValue(img),1.5)+1, img->n, pathval->val);

  for (p = 0; p < img->n; p++)
  {
    pathval->val[p] = IFT_INFINITY_INT;
  }

  while (S != NULL)
  {
    p = S->elem;
    label->val[p]   = S->label;
    pathval->val[p] = 0;
    iftInsertGQueue(&Q, p);
    S = S->next;
  }

  // Image Foresting Transform

  while (!iftEmptyGQueue(Q))
  {
    p = iftRemoveGQueue(Q);
    u = iftGetVoxelCoord(img, p);

    for (i = 1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A, u, i);

      if (iftValidVoxel(img, v))
      {
        q = iftGetVoxelIndex(img, v);

	if (Q->L.elem[q].color != IFT_BLACK) {

	    float Wi = alpha*gradI->val[q] + (1-alpha)*gradO->val[q];
	//   float Wi = alpha*gradI->val[q] + (1-alpha)*featureEuclidianDistance(feature_vector, p, q);// featureEuclidianDistance(feature_vector->val[p], feature_vector->val[q], feature_vector->m);
          tmp = iftMax(pathval->val[p], iftRound(Wi));

          if (tmp < pathval->val[q])  {
	    if (Q->L.elem[q].color == IFT_GRAY)
	      iftRemoveGQueueElem(Q,q);
            label->val[q]    = label->val[p];
            pathval->val[q]  = tmp;
            iftInsertGQueue(&Q, q);
          }
        }
      }
    }
  }
  iftDestroyAdjRel(&A);
  iftDestroyGQueue(&Q);
  iftDestroyLabeledSet(&S);
  iftDestroyImage(&gradO);
  iftDestroyImage(&fimg);
  iftDestroyImage(&gradI);
  iftDestroyImage(&pathval);
  iftCopyVoxelSize(img, label);

  return (label);
}







int main(int argc, char *argv[]) 
{
  timer *tstart = NULL;
  int    MemDinInicial, MemDinFinal;
  
  MemDinInicial = iftMemoryUsed(1);

  if (argc != 6){
    iftError("Usage: watershed <...>\n"
	     "[1] input image .scn \n"
	     "[2] input labeled seeds .txt  \n"
	     "[3] input object feature vector (.mimg) \n"
	     "[4] feature vector model (.txt) \n"
	     "[5] output label image .scn \n",
	     "main");
  }

  tstart = iftTic();

  /* ----------------------- Coding Area -------------------------- */

  iftImage *img             = iftReadImageByExt(argv[1]);
  iftLabeledSet *S          = iftReadSeeds(img,argv[2]);
  iftMImage *feature_vector = iftReadMImage(argv[3]);

  FILE *fp_model = fopen(argv[4], "r");
  int n;
  int *filters;

  fscanf(fp_model, "%d\n", &n);
  filters = (int *)calloc(n, sizeof(int));

  for (int i = 0; i < n; i++){
    fscanf(fp_model, "%d ", &filters[i]);
  }

  // Filter feature vector according to model
  iftMImage *filtered_feature_vector = iftCreateMImage(feature_vector->xsize, feature_vector->ysize, feature_vector->zsize, n);
  
  for (int i = 0; i < n; i++){
    for (int p = 0; p < filtered_feature_vector->n; p++){
        filtered_feature_vector->val[p][i] = feature_vector->val[p][filters[i]];
    }
  }
  
  iftImage *label  = SegmentByWatershed(img, S, filtered_feature_vector);

  iftWriteImageByExt(label,argv[5]);

  iftDestroyMImage(&feature_vector);
  iftDestroyMImage(&filtered_feature_vector);
  iftDestroyImage(&img);
  iftDestroyImage(&label);
  iftDestroyLabeledSet(&S);

  free(filters);

  
  /* -------------------- End of the coding area ----------------- */
    
  puts("\nDone...");
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
    
  MemDinFinal = iftMemoryUsed();
  iftVerifyMemory(MemDinInicial, MemDinFinal);
  
  return(0);
}
