#ifndef DESTRUCT_H
#define DESTRUCT_H

#include <stdlib.h>
#include "Init.h"

void destruct(
    particle p, long double **times
) {

    free(p.x);
    free(p.px);
    free(p.q);
    free(p.m);
    
    free(*times);

}
#endif
