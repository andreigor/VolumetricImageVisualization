#include <stdio.h>
#include <stdlib.h>

#include "ift.h"


int main(int argc, char *argv[]){
    timer *tstart = NULL;
    int    MemDinInicial, MemDinFinal;

    MemDinInicial = iftMemoryUsed(1);

    if ((argc != 3)){
        printf("usage segmentation_report: <P1> <P2>\n");
        printf("P1: predicted mask (.scn)\n");
        printf("P2: segmentation label (.scn)\n");
        exit(0);
    }
    tstart = iftTic();

    iftImage *predicted = iftReadImageByExt(argv[1]);
    iftImage *label     = iftReadImageByExt(argv[2]);


    double dice = iftDiceSimilarity(predicted, label);

    printf("Dice similarity: %lf\n", dice);


    iftDestroyImage(&predicted);
    iftDestroyImage(&label);

    

    /* -------------------- End of the coding area ----------------- */
        
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
        
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    
    return(0);
}