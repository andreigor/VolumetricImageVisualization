#include "ift.h"
  
iftImage *SegmentByWatershed(iftImage *img, iftLabeledSet *seeds, iftImage *omap, float alpha)
{
  iftImage   *pathval = NULL, *label = NULL, *gradI=NULL, *gradO=NULL;
  iftGQueue  *Q = NULL;
  int            i, p, q, tmp;
  iftVoxel       u, v;
  iftLabeledSet *S = seeds;
  iftAdjRel     *A = iftSpheric(1.0);

  // Initialization
  
  pathval    = iftCreateImage(img->xsize, img->ysize, img->zsize);
  label      = iftCreateImage(img->xsize, img->ysize, img->zsize);
  gradI      = iftImageGradientMagnitude(img,A);
  gradO      = iftImageGradientMagnitude(omap,A);
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

	  float Wi = alpha*gradI->val[q]+(1.0-alpha)*gradO->val[p];
	  
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
  iftDestroyImage(&gradO);
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
	     "[3] input object map .scn \n"
	     "[4] input alpha [0,1] \n"
	     "[5] output label image .scn \n",
	     "main");
  }

  tstart = iftTic();

  /* ----------------------- Coding Area -------------------------- */

  iftImage *img    = iftReadImageByExt(argv[1]);
  iftLabeledSet *S = iftReadSeeds(img,argv[2]);
  iftImage *omap   = iftReadImageByExt(argv[3]);
  float alpha      = atof(argv[4]);
  iftImage *label  = SegmentByWatershed(img, S, omap, alpha);

  iftWriteImageByExt(label,argv[5]);

  iftDestroyImage(&omap);
  iftDestroyImage(&img);
  iftDestroyImage(&label);
  iftDestroyLabeledSet(&S);

  
  /* -------------------- End of the coding area ----------------- */
    
  puts("\nDone...");
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
    
  MemDinFinal = iftMemoryUsed();
  iftVerifyMemory(MemDinInicial, MemDinFinal);
  
  return(0);
}
