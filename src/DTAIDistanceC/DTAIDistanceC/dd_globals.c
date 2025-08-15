//
//  dd_globals.c
//  DTAIDistanceC
//
//  Created by Wannes Meert on 24/08/2020.
//  Copyright © 2020 Wannes Meert. All rights reserved.
//

#include <stdio.h>
#include "dd_ed.h"

void print_precision_set(int precision) {
    printPrecision = precision;
}

void print_precision_reset(void) {
    printPrecision = 3;
}

void print_nb(seq_t value) {
    snprintf(printFormat, sizeof(printFormat), "%%.%df", printPrecision);
    snprintf(printBuffer, sizeof(printBuffer), printFormat, value);
    printf("%*s", printDigits, printBuffer);
    // "%-*s" would left align
}

void print_ch(char* string) {
    printf("%*s", printDigits, string);
    // "%-*s" would left align
}
