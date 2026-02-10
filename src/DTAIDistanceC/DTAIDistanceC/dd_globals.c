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

void print_nbs(seq_t* values, idx_t b, idx_t e) {
    printf("[");
    for (idx_t a=b; a<e; a++) {
        print_nb(values[a]);printf(",");
    }
    printf("]\n");
}

void print_ch(char* string) {
    printf("%*s", printDigits, string);
    // "%-*s" would left align
}

// MARK: Path

void dd_path_init(DDPath *a, idx_t initial_capacity) {
    if (initial_capacity > 0) {
        a->array = malloc(initial_capacity * sizeof(DDPathEntry));
        if (a->array == NULL) {
            printf("ERROR: cannot allocate memory for storing the path");
            exit(1);
        }
    } else {
        a->array = NULL;
    }
    a->length = 0;
    a->capacity = initial_capacity;
    a->distance = 0;
}

void dd_path_insert(DDPath *a, idx_t i, idx_t j) {
    DDPathEntry* array_new;
    if (a->length == a->capacity) {
        if (a->array == NULL) {
            dd_path_init(a, 2);
        } else {
            assert(a->capacity > 0);
            a->capacity *= 2;
            array_new = realloc(a->array, a->capacity * sizeof(DDPathEntry));
            if (array_new == NULL) {
                printf("ERROR: cannot allocate memory for storing the path");
                if (a->array != NULL ){
                    free(a->array);
                    a->array = NULL;
                }
                exit(1);
                return;
            } else {
                a->array = array_new;
            }
        }
    }
    a->array[a->length++] = (DDPathEntry){.i=i, .j=j};
}

void dd_path_insert_wo_doubles(DDPath *a, idx_t i, idx_t j) {
    if (a->length > 0 && a->array[a->length-1].i == i && a->array[a->length-1].j == j) {
        return;
    }
    dd_path_insert(a, i, j);
}

void dd_path_extend(DDPath *a, DDPath *b) {
    DDPathEntry entry;
    for (int i=0; i<b->length; i++) {
        entry = b->array[i];
        dd_path_insert(a, entry.i, entry.j);
    }
}

void dd_path_extend_wo_doubles(DDPath *a, DDPath *b, idx_t overlap) {
    DDPathEntry entry;
    if (overlap > b->length) {
        overlap = b->length;
    }
    for (idx_t i=0; i<overlap; i++) {
        entry = b->array[i];
        dd_path_insert_wo_doubles(a, entry.i, entry.j);
    }
    for (idx_t i=overlap; i<b->length; i++) {
        entry = b->array[i];
        dd_path_insert(a, entry.i, entry.j);
    }
}

void dd_path_extend_wo_overlap(DDPath *a, DDPath *b, idx_t overlap) {
    DDPathEntry entry;
    for (idx_t i=overlap; i<b->length; i++) {
        entry = b->array[i];
        dd_path_insert(a, entry.i, entry.j);
    }
}

void dd_path_free(DDPath *a) {
    free(a->array);
    a->array = NULL;
    a->length = a->capacity = 0;
    a->distance = 0;
}

void dd_path_reverse(DDPath *a) {
    DDPathEntry temp;
    for (int i=0; i<a->length/2; i++) {
        temp = a->array[i];
        a->array[i] = a->array[a->length-1-i];
        a->array[a->length-1-i] = temp;
    }
}

void dd_path_print(DDPath *path) {
    DDPathEntry *entry;
    printf("[");
    for (int i=0; i<path->length; i++) {
        entry = &path->array[i];
        if (i!=0) {
            printf(",");
        }
        printf("(%zu,%zu)", entry->i, entry->j);
    }
    printf("]\n");
}
