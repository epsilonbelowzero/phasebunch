#ifndef DESTRUCT_H
#define DESTRUCT_H

#include <stdlib.h>
#include "Init.h"

void destruct(
    particle p
) {

    free(p.x);
    free(p.px);
    free(p.q);
    free(p.m);
    
    cudaFree(p.dev_x);
    cudaFree(p.dev_px);
    cudaFree(p.dev_m);
    cudaFree(p.dev_q);
}
#endif
